"""Week-one rarity forecast for GTA VI trophies (make.fun gta6-trophy-rarity-forecast-01).

Idea, in one paragraph: every trophy's earn rate is modelled on the logit
scale as a roughly straight line in log-time since launch (19 Nov 2026
00:00Z). A weighted least-squares slope is fitted per trophy from the reads
that actually contain it, recent reads count more, and a read that follows a
sharp jump in players_tracked counts less, because fresh players enter the
tracked pool before they earn anything, so the rate dips for accounting
reasons, not because players stopped earning. The fitted slope is shrunk
toward a prior slope for the trophy's kind, and the level is pulled toward a
prior adoption curve for that kind; the pull grows with the horizon, so a
90-minute practice case stays close to the last read while a 4.5-day final
case leans on what each kind of trophy can plausibly do in a week. Spreads
are honest: they grow with the square root of the horizon plus how
unidentified the slope is (a frozen history tells us almost nothing about
the slope), so the 90% band is wide when it should be.

Only the standard library is used. No network, no randomness, no wall-clock
dependence: the same input bytes always produce the same output bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

ASKED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Per-kind prior: (ceiling %, time constant hours, prior slope in logit per
# unit of log-time, extra spread, max pull toward the prior curve).
# Ceilings are week-one plausibility caps, not lifetime rarities: chapter 1
# is finishable in an evening, 100% Completion is not finishable in a week
# by more than a rounding error of players, and the platinum needs all of it.
KINDS = {
    "platinum":    (0.25, 500.0, 0.05, 0.25, 0.92),
    "completion":  (0.40, 500.0, 0.05, 0.25, 0.90),
    "grind_hard": (2.0, 400.0, 0.25, 0.20, 0.60),
    "grind_mid":  (8.0, 200.0, 0.45, 0.15, 0.55),
    "story1":     (60.0, 30.0, 1.20, 0.10, 0.65),
    "story2":     (50.0, 48.0, 1.05, 0.10, 0.65),
    "story3":     (40.0, 72.0, 0.90, 0.10, 0.65),
    "story4":     (32.0, 96.0, 0.80, 0.10, 0.65),
    "story_final": (22.0, 144.0, 0.70, 0.12, 0.65),
    "misc":       (25.0, 72.0, 0.70, 0.12, 0.50),
}

# Hard upper bound on the median, per kind (percent). A safety rail so a
# noisy positive slope can never push the platinum, or 100% Completion,
# anywhere near the frozen early rate of 0.63%.
MEDIAN_CAP = {
    "platinum": 1.0,
    "completion": 1.5,
    "grind_hard": 6.0,
    "grind_mid": 20.0,
    "story_final": 40.0,
    "story4": 55.0,
    "story3": 65.0,
    "story2": 75.0,
    "story1": 85.0,
    "misc": 50.0,
}

GRIND_HINTS = (
    "50 challenges", "collectible card", "100 locations",
    "8 mission strands", "5 bank vaults", "100 vehicles",
    "20 different races", "meditate at every", "drone race at every",
)


def parse_moment(text):
    """ISO-8601 timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def hours_after_launch(moment):
    """Hours since launch; pre-launch reads count as hour zero."""
    return max(0.0, (moment - LAUNCH).total_seconds() / 3600.0)


def log_clock(hours):
    """Log-time coordinate: ln(1 + hours since launch)."""
    return math.log1p(hours)


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def pct_to_logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def logit_to_pct(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    top = math.exp(value)
    return 100.0 * top / (1.0 + top)


def four_decimals(value):
    """Round to 4 decimals, halves away from zero (up for positives)."""
    stepped = math.floor(value * 10000.0 + 0.5) / 10000.0
    return stepped


# --- Standard normal quantile (Acklam's rational approximation). ---

_NORM_A = (-3.969683028665376e01, 2.209460984245205e02,
           -2.759285104469687e02, 1.383577518672690e02,
           -3.066479806614716e01, 2.506628277459239e00)
_NORM_B = (-5.447609879822406e01, 1.615858368580409e02,
           -1.556989798598866e02, 6.680131188771972e01,
           -1.328068155288572e01)
_NORM_C = (-7.784894002430293e-03, -3.223964580411365e-01,
           -2.400758277161838e00, -2.549732539343734e00,
           4.374664141464968e00, 2.938163982698783e00)
_NORM_D = (7.784695709041462e-03, 3.224671290700398e-01,
           2.445134137142996e00, 3.754408661907416e00)


def gauss_quantile(prob):
    """Approximate inverse of the standard normal CDF at prob in (0, 1)."""
    if prob <= 0.0 or prob >= 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    split_low = 0.02425
    split_high = 1.0 - split_low
    if prob < split_low:
        q = math.sqrt(-2.0 * math.log(prob))
        top = (((((_NORM_C[0] * q + _NORM_C[1]) * q + _NORM_C[2]) * q
                 + _NORM_C[3]) * q + _NORM_C[4]) * q + _NORM_C[5])
        bottom = (((_NORM_D[0] * q + _NORM_D[1]) * q + _NORM_D[2]) * q
                  + _NORM_D[3]) * q + 1.0
        return top / bottom
    if prob > split_high:
        q = math.sqrt(-2.0 * math.log(1.0 - prob))
        top = (((((_NORM_C[0] * q + _NORM_C[1]) * q + _NORM_C[2]) * q
                 + _NORM_C[3]) * q + _NORM_C[4]) * q + _NORM_C[5])
        bottom = ((((_NORM_D[0] * q + _NORM_D[1]) * q + _NORM_D[2]) * q
                   + _NORM_D[3]) * q + 1.0)
        return -top / bottom
    q = prob - 0.5
    r = q * q
    top = (((((_NORM_A[0] * r + _NORM_A[1]) * r + _NORM_A[2]) * r
             + _NORM_A[3]) * r + _NORM_A[4]) * r + _NORM_A[5]) * q
    bottom = (((((_NORM_B[0] * r + _NORM_B[1]) * r + _NORM_B[2]) * r
                + _NORM_B[3]) * r + _NORM_B[4]) * r + 1.0)
    return top / bottom


def kind_of(trophy):
    """Classify a trophy from its tier, name and description."""
    tier = str(trophy.get("tier", "")).lower()
    text = (str(trophy.get("name", "")) + " " +
            str(trophy.get("description", ""))).lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in text:
        return "completion"
    if "250 stunts" in text or "shooting range" in text:
        return "grind_hard"
    if "chapter 1" in text:
        return "story1"
    if "chapter 2" in text:
        return "story2"
    if "chapter 3" in text:
        return "story3"
    if "chapter 4" in text:
        return "story4"
    if "final mission" in text:
        return "story_final"
    if any(hint in text for hint in GRIND_HINTS):
        return "grind_mid"
    if tier == "gold":
        return "grind_mid"
    return "misc"


def prior_pct(kind, target_hours):
    """Prior adoption curve: ceiling * (1 - exp(-hours / tau))."""
    ceiling, tau, _slope, _extra, _pull = KINDS[kind]
    if target_hours <= 0.0:
        return 0.01
    return max(0.01, ceiling * (1.0 - math.exp(-target_hours / tau)))


def tracked_jump(reads, index):
    """True when this read's tracked count jumped >25% over the last read."""
    if index <= 0:
        return False
    before = reads[index - 1].get("players_tracked")
    now = reads[index].get("players_tracked")
    if not before or not now:
        return False
    return now > 1.25 * before


def fit_line(points):
    """Weighted least squares y = level + slope * (x - x_last).

    points: list of (x, y, weight). Returns (level, slope, resid_rms).
    With one usable point the slope is zero and the level is that point.
    """
    if not points:
        return None, 0.0, None
    x_last = points[-1][0]
    total_w = sum(w for _x, _y, w in points)
    if total_w <= 0:
        return points[-1][1], 0.0, None
    mean_x = sum(w * (x - x_last) for x, _y, w in points) / total_w
    mean_y = sum(w * y for _x, y, w in points) / total_w
    num = sum(w * ((x - x_last) - mean_x) * (y - mean_y)
              for x, y, w in points)
    den = sum(w * ((x - x_last) - mean_x) ** 2 for x, _y, w in points)
    slope = num / den if den > 1e-12 else 0.0
    level = mean_y - slope * mean_x
    resid = None
    if len(points) >= 3 and den > 1e-12:
        sq = sum(w * (y - (level + slope * (x - x_last))) ** 2
                 for x, y, w in points)
        resid = math.sqrt(sq / total_w)
    return level, slope, resid


def forecast_trophy(trophy, reads, target_hours, horizon_days):
    """One trophy's five quantiles."""
    kind = kind_of(trophy)
    _ceiling, _tau, prior_slope, extra_spread, max_pull = KINDS[kind]
    idx = trophy["_pos"]

    points = []
    for j, read in enumerate(reads):
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        rate = float(rates[idx])
        try:
            moment = parse_moment(read.get("at") or read.get("slot"))
        except (TypeError, ValueError):
            continue
        weight = 0.8 ** (len(reads) - 1 - j)
        if tracked_jump(reads, j):
            prev_rates = reads[j - 1].get("rates") or []
            prev = prev_rates[idx] if idx < len(prev_rates) else None
            if prev is not None and rate < float(prev):
                # A refresh dip: new players dilute the rate for accounting
                # reasons only, and one such point would drag the fitted
                # slope negative. Drop it from the fit entirely; the reads
                # after it (no jump) still enter normally.
                continue
        points.append((log_clock(hours_after_launch(moment)),
                       pct_to_logit(rate), weight))

    # Distinct levels among the points actually kept: a dropped refresh dip
    # must not count as "movement" that would dilute the prior slope.
    distinct_rates = {round(y, 6) for _x, y, _w in points}

    # A trophy everyone already has stays earned: pin it, tight band.
    last_rate = None
    if reads:
        last_rates = reads[-1].get("rates") or []
        if idx < len(last_rates) and last_rates[idx] is not None:
            last_rate = float(last_rates[idx])
    if last_rate is not None and last_rate >= 99.0:
        centre = pct_to_logit(99.9)
        sigma = 0.08
        return [four_decimals(clamp_pct(logit_to_pct(centre + gauss_quantile(lv) * sigma)))
                for lv in ASKED_LEVELS]

    informative = max(0, len(distinct_rates) - 1)
    level, slope, resid = fit_line(points)
    x_last = points[-1][0] if points else log_clock(hours_after_launch(LAUNCH))
    x_target = log_clock(target_hours)

    # Anchor: if the last read is a refresh dip, average the fitted level
    # with the previous read's logit so one dilution event cannot drag the
    # forecast down on its own.
    if points and len(points) >= 2 and tracked_jump(reads, len(reads) - 1):
        prev_rates = reads[-2].get("rates") or []
        prev = prev_rates[idx] if idx < len(prev_rates) else None
        if prev is not None and last_rate is not None and last_rate < float(prev):
            level = 0.5 * level + 0.5 * pct_to_logit(float(prev))

    if level is None:
        # Never observed: fall back to the prior curve alone, wide band.
        centre = pct_to_logit(prior_pct(kind, target_hours))
        sigma = 1.5
        return quantiles_from(centre, sigma)

    # Shrink the fitted slope toward the kind's prior slope. A frozen
    # history (one distinct rate) carries no slope information at all, so
    # the prior slope takes over completely; real movement earns weight.
    shrink_k = 3.0
    slope_star = (informative * slope + shrink_k * prior_slope) / (informative + shrink_k)
    slope_star = min(2.0, max(-0.15, slope_star))
    step = max(0.0, x_target - x_last)
    centre = level + slope_star * step

    # Pull the level toward the prior adoption curve. The pull grows with
    # the horizon: negligible at 90 minutes, dominant over 4.5 days.
    grow = 1.0 - math.exp(-horizon_days / 2.0)
    pull = max_pull * grow
    if last_rate is not None and last_rate > 50.0:
        pull *= 0.3
    prior_centre = pct_to_logit(prior_pct(kind, target_hours))
    centre = (1.0 - pull) * centre + pull * prior_centre
    centre = min(centre, pct_to_logit(MEDIAN_CAP[kind]))

    # Spread: base + random-walk growth + slope ignorance + instability.
    sigma = 0.30 + 0.35 * math.sqrt(horizon_days)
    sigma += 0.55 / (1.0 + informative)
    sigma += extra_spread
    if resid is not None:
        sigma += 0.30 * min(resid, 1.0)
    if len(reads) >= 2 and tracked_jump(reads, len(reads) - 1):
        sigma += 0.25
    sigma = min(2.2, max(0.35, sigma))
    return quantiles_from(centre, sigma)


def quantiles_from(centre, sigma, levels=None):
    levels = levels if levels is not None else ASKED_LEVELS
    out = []
    prev = None
    for lv in levels:
        value = four_decimals(clamp_pct(logit_to_pct(centre + gauss_quantile(lv) * sigma)))
        if prev is not None and value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def forecast_case(case):
    trophies = case.get("trophies") or []
    reads = case.get("history") or []
    horizon = float(case.get("horizon_minutes") or 0)
    horizon_days = max(0.0, horizon / 1440.0)
    try:
        target_hours = hours_after_launch(parse_moment(case["target"]))
    except (KeyError, TypeError, ValueError):
        last = reads[-1] if reads else None
        base = parse_moment(last.get("at")) if last else LAUNCH
        target_hours = hours_after_launch(base) + horizon / 60.0
    rows = []
    for pos, trophy in enumerate(trophies):
        marked = dict(trophy)
        marked["_pos"] = pos
        try:
            qs = forecast_trophy(marked, reads, target_hours, horizon_days)
        except Exception:
            kind = kind_of(marked)
            qs = quantiles_from(pct_to_logit(prior_pct(kind, target_hours)), 1.5)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
