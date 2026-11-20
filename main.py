#!/usr/bin/env python3
"""GTA VI trophy rarity forecast, v2.

Model: earners-vs-players dilution.  A trophy's earn rate is a ratio,
rate = 100 * E / T, with E = players who earned it and T = players
Exophase tracks.  The two sides move for different reasons -- players
earn trophies over time, while Exophase tracks more and more players who
have mostly earned nothing yet -- so fitting the rate directly is wrong:
after a refresh the tracked count jumps and every rate DILUTES down,
even though earners only grew.  v1 fitted the rate and predicted UP;
v2 fits both sides separately and divides.

Design, step by step:

1. DECOMPOSITION.  For every trophy and every read, E = rate * T / 100
   (float) and T = players_tracked.  Work in logs: log(rate) = log(E) -
   log(T).  Fit TWO lines against x = ln(1 + hours since launch), one
   for log(E) and one for log(T), each by weighted least squares with
   recency weights 0.8**age_hours (age = hours from the read to the
   checkpoint).  Extrapolate both lines to the target and take
   rate_target = 100 * E_target / T_target, clamped to 0.01..99.99.
   Then cap the median per trophy KIND (see CAPS below).

2. KIND CLASSIFICATION.  A trophy's kind is decided from its own data
   (tier + name + description) by classify(), and the rules are written
   out there in comments.  In short:
     platinum   = tier platinum.
     completion = attains 100% completion.
     story_final = completes the final mission (exactly that, not a
       side character's "final mission for X").
     story1..4 = "Complete Chapter N." (N = 1..4, earliest to latest).
     grind_hard = heavy repetition / collection (250 stunts, 100
       locations, 100 vehicles, 50 challenges, 3 card sets, gold in
       every shooting challenge, 20 races, every drone-race location,
       every meditation spot, all garage properties, 8 stranger
       strands, explore all of Leonida).
     grind_mid = lighter repetition / collection (5 bank vaults, 10
       vehicles, try each activity once, an arcade game, beat Cal,
       $250k laundering, $100k spending, a unique-animal hunt, a side
       character's final mission).
     misc       = everything else; when in doubt, misc.  Named story
       missions that are not numbered chapters ("Complete Breakfast
       Run." etc.) go to misc because their place in the story order
       cannot be told from the data.

3. STALE / FROZEN DETECTION.  If every read in the history is IDENTICAL
   (same rates and same tracked count), those values are one STALE
   snapshot from Exophase's last refresh before the first read, not a
   series of fresh measurements.  The fit is then anchored at the FIRST
   read (when the snapshot was actually measured, ~launch), never at
   the last read: the frozen reads after it add no information and must
   not drag the anchor to "now".  When the history varies, the anchor
   is the last read, as usual.

4. PRIOR SLOPES (used when history is frozen / a slope is unidentified).
   Earners grow SLOWLY in the launch window (new tracked players have
   earned nothing yet): platinum 0.02, completion 0.02, grind_hard
   0.04, grind_mid 0.075, story1 0.15, story2 0.12, story3 0.10,
   story4 0.08, story_final 0.06, misc 0.10 (all in x = ln(1+hours)
   units).  Tracked grows FAST at first and slows:
   slope_T_prior(t) = 0.25 + 0.85 * exp(-t_hours / 36), with t_hours =
   hours from launch to the checkpoint (about 1.00 at 4.5 h, 0.29 at
   108 h).  Shrinkage: slope* = (info * slope_fit + 3 * prior) /
   (info + 3), where info = distinct observed values - 1 (frozen ->
   info 0 -> full prior).  Earners slope clamped to [-0.2, 2.0],
   tracked slope to [0, 2.0].

5. SPREAD (honest and wide enough).  sigma = 0.35 + 0.30 *
   sqrt(horizon_days) + frozen_bonus + 0.20 * min(resid_E, 1) +
   0.20 * min(resid_T, 1), clamped to [0.4, 2.4]; frozen_bonus is
   0.30 when frozen, else 0.30 / (1 + info); resid_E / resid_T are the
   standard deviations of the log(E) / log(T) fit residuals.
   Quantiles come from logit(median) + z_tau * sigma with
   z = (-1.645, -0.674, 0, 0.674, 1.645), mapped back to percent,
   rounded to 4 decimals, forced monotone non-decreasing.

6. EDGE CASES.  Rate 100% clamps to a 99.99 logit; rate 0.00% clamps
   to 0.01; a trophy whose last rate is >= 99% stays pinned at 99.9
   with a narrow band (sigma 0.08); reads with a null rate for a
   trophy are skipped for that trophy; a trophy never observed at all
   gets a prior-only forecast (median 0.5% capped by its kind cap)
   with a wide band (sigma 1.5).

Runs with no network, stdlib only.  Reads $MAKEFUN_INPUT_DIR (else
/input/), writes $MAKEFUN_OUTPUT_DIR (else /output/), one *.out per
*.in found, in name order.  Deterministic: no randomness anywhere.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone

# Standard-normal quantiles for the five levels 0.05/0.25/0.5/0.75/0.95.
Z_TAU = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)

# Median caps per kind, in percent.  Rare trophies cannot plausibly be
# common a week after launch; the caps stop the dilution division from
# ever claiming they are.
CAPS = {
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

# Prior earners slope per kind (log E per unit of ln(1 + hours)).
# Small in the launch window: fresh tracked players dilute first.
EARN_PRIOR = {
    "platinum": 0.02,
    "completion": 0.02,
    "grind_hard": 0.04,
    "grind_mid": 0.075,
    "story1": 0.15,
    "story2": 0.12,
    "story3": 0.10,
    "story4": 0.08,
    "story_final": 0.06,
    "misc": 0.10,
}

# Phrases (lowercased name + description) marking heavy grind trophies.
_GRIND_HARD = (
    "250 stunts",
    "100 locations",
    "100 vehicles",
    "50 challenges",
    "collectible card sets",
    "gold medal in all",
    "20 different races",
    "every location",
    "every spot",
    "all garage properties",
    "explore all of",
    "8 mission strands",
    "meditate at every",
)

# Phrases marking lighter grind / side-content trophies.
_GRIND_MID = (
    "5 bank vaults",
    "10 vehicles",
    "try each type",
    "arcade game",
    "beat cal",
    "earn $250,000",
    "spend $100,000",
    "hunt a unique animal",
    "final mission for ",
)


def classify(trophy):
    """Kind of one trophy, from its own tier + name + description.

    Order matters: platinum and completion first, then the story
    spine (final mission, then numbered chapters), then grind by
    size of the ask, then misc as the fallback for anything unclear.
    Named story missions without a chapter number ("Complete Breakfast
    Run.") fall through to misc -- their order in the story cannot be
    told from the data, and guessing would mis-cap them.
    """
    tier = (trophy.get("tier") or "").lower()
    name = trophy.get("name") or ""
    desc = trophy.get("description") or ""
    text = (name + " " + desc).lower()
    desc_l = desc.lower()

    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc_l:
        return "completion"
    # "Complete the final mission." only: a side character's
    # "final mission for Andy/Kenton/..." is NOT the story finale.
    if "final mission" in desc_l and " for " not in desc_l:
        return "story_final"
    m = re.search(r"complete chapter (\d+)", desc_l)
    if m:
        chapter = int(m.group(1))
        if chapter <= 1:
            return "story1"
        if chapter == 2:
            return "story2"
        if chapter == 3:
            return "story3"
        return "story4"
    if any(p in text for p in _GRIND_HARD):
        return "grind_hard"
    if any(p in text for p in _GRIND_MID):
        return "grind_mid"
    return "misc"


def _parse_time(s):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def _clamp_pct(x):
    return min(99.99, max(0.01, float(x)))


def _logit_pct(x):
    """logit of a percent, with 0% / 100% clamped first."""
    p = _clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit_pct(logit):
    """Percent from a logit value, clamped to 0.01..99.99."""
    if logit >= 0.0:
        p = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        p = e / (1.0 + e)
    return min(99.99, max(0.01, 100.0 * p))


def _wls_line(xs, ys, ws):
    """Weighted least squares y = a + b*x; None if slope unidentified."""
    if len(xs) < 2:
        return None
    sw = sum(ws)
    if sw <= 0.0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den <= 0.0:
        return None
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / den
    return (my - b * mx, b)


def _resid_std(xs, ys, line):
    """Population std of the residuals of a fitted line (0 if no fit)."""
    if line is None or len(xs) < 2:
        return 0.0
    a, b = line
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    mean = sum(res) / len(res)
    return math.sqrt(sum((r - mean) ** 2 for r in res) / len(res))


def _shrink(fit, prior, info):
    """Shrink a fitted slope toward its prior; info 0 -> full prior."""
    if fit is None:
        return prior
    return (info * fit + 3.0 * prior) / (info + 3.0)


def _quantiles(median, sigma):
    """Five quantiles around a median percent, monotone, 4 decimals."""
    centre = _logit_pct(median)
    out = []
    for z in Z_TAU:
        out.append(round(_inv_logit_pct(centre + z * sigma), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_trophy(trophy, idx, history, launch, checkpoint, target,
                    horizon_days):
    """Forecast one trophy: (median percent, sigma) -> five quantiles."""
    kind = classify(trophy)
    cap = CAPS[kind]

    # Collect one observation per read: (x, logE | None, logT, weight,
    # rate | None, tracked).  x = ln(1 + hours since launch); weight is
    # recency 0.8**age_hours.  A null rate skips logE for that read only.
    points = []
    for read in history:
        tracked = read.get("players_tracked")
        if not tracked or tracked <= 0:
            continue
        rates = read.get("rates") or []
        rate = rates[idx] if idx < len(rates) else None
        when = _parse_time(read["at"])
        hours = max(0.0, (when - launch).total_seconds() / 3600.0)
        age = max(0.0, (checkpoint - when).total_seconds() / 3600.0)
        log_e = None
        if rate is not None:
            earners = max(float(rate), 0.01) * float(tracked) / 100.0
            log_e = math.log(max(earners, 1e-9))
        points.append((math.log1p(hours), log_e, math.log(float(tracked)),
                       0.8 ** age, rate, float(tracked)))

    # Never observed: prior curve only, wide band.
    observed = [p for p in points if p[1] is not None]
    if not observed:
        return _quantiles(min(0.5, cap), 1.5)

    last_rate = observed[-1][4]
    # A trophy (nearly) everyone tracked has earned stays pinned at the
    # top with a narrow band instead of being diluted down with the rest.
    if last_rate is not None and last_rate >= 99.0:
        return _quantiles(99.9, 0.08)

    # Frozen = every read identical (same rates, same tracked): one stale
    # snapshot, not a series.  Anchor it at the FIRST read, when the
    # snapshot was actually measured; varying history anchors at the
    # LAST read.  info = distinct values - 1, so frozen -> info 0 and
    # the prior slopes take over completely.
    rates_seen = [p[4] for p in observed]
    tracked_seen = [p[5] for p in points]
    frozen = len(set(rates_seen)) <= 1 and len(set(tracked_seen)) <= 1
    info_e = len({p[1] for p in observed}) - 1
    info_t = len({p[2] for p in points}) - 1
    if frozen:
        info_e = 0
        info_t = 0

    if frozen:
        anchor_e = observed[0]
        anchor_t = points[0]
    else:
        anchor_e = observed[-1]
        anchor_t = points[-1]

    # Fit log(E) and log(T) against x by recency-weighted least squares.
    ex = [p[0] for p in observed]
    ey = [p[1] for p in observed]
    ew = [p[3] for p in observed]
    tx = [p[0] for p in points]
    ty = [p[2] for p in points]
    tw = [p[3] for p in points]
    line_e = _wls_line(ex, ey, ew)
    line_t = _wls_line(tx, ty, tw)
    resid_e = _resid_std(ex, ey, line_e)
    resid_t = _resid_std(tx, ty, line_t)

    # Priors.  Tracked prior depends on how far past launch the
    # checkpoint is: fast early, slowing down.
    t_cp = max(0.0, (checkpoint - launch).total_seconds() / 3600.0)
    prior_t = 0.25 + 0.85 * math.exp(-t_cp / 36.0)
    prior_e = EARN_PRIOR[kind]
    slope_e = _shrink(line_e[1] if line_e else None, prior_e, info_e)
    slope_t = _shrink(line_t[1] if line_t else None, prior_t, info_t)
    slope_e = min(2.0, max(-0.2, slope_e))
    slope_t = min(2.0, max(0.0, slope_t))

    # Extrapolate both sides from the anchor to the target, then divide.
    x_target = math.log1p(max(0.0, (target - launch).total_seconds()
                              / 3600.0))
    log_e_t = anchor_e[1] + slope_e * (x_target - anchor_e[0])
    log_t_t = anchor_t[2] + slope_t * (x_target - anchor_t[0])
    gap = min(4.6, max(-9.21, log_e_t - log_t_t))
    median = min(99.99, max(0.01, 100.0 * math.exp(gap)))
    median = min(median, cap)

    # Spread: base + horizon growth + frozen bonus + fit residuals.
    if frozen:
        frozen_bonus = 0.30
    else:
        frozen_bonus = 0.30 / (1.0 + info_e)
    sigma = (0.35 + 0.30 * math.sqrt(max(0.0, horizon_days))
             + frozen_bonus
             + 0.20 * min(resid_e, 1.0) + 0.20 * min(resid_t, 1.0))
    sigma = min(2.4, max(0.4, sigma))

    return _quantiles(median, sigma)


def _launch_time(case):
    """Launch = release_date at 00:00 UTC; fallback: earliest read."""
    game = case.get("game") or {}
    rel = game.get("release_date")
    if rel:
        try:
            dt = datetime.fromisoformat(str(rel).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    history = case.get("history") or []
    if history:
        return _parse_time(history[0]["at"])
    return _parse_time(case["at"])


def process_case(case):
    """One case dict -> {"trophies": [{"id": n, "q": [...]}]}."""
    launch = _launch_time(case)
    checkpoint = _parse_time(case["at"])
    horizon_min = float(case.get("horizon_minutes") or 0)
    target_raw = case.get("target")
    if target_raw:
        target = _parse_time(target_raw)
    else:
        target = checkpoint + timedelta(minutes=horizon_min)
    horizon_days = horizon_min / 1440.0

    trophies = case.get("trophies") or []
    history = case.get("history") or []
    rows = []
    for idx, trophy in enumerate(trophies):
        q = forecast_trophy(trophy, idx, history, launch, checkpoint,
                            target, horizon_days)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        result = process_case(case)
        stem = case.get("id") or name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
