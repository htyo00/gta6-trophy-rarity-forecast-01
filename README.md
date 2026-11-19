# gta6-trophy-rarity-forecast-01

Forecast how many GTA VI players earn each trophy in week one, from
Exophase rarity reads (make.fun competition `gta6-trophy-rarity-forecast-01`).

## Program

`main.py` — Python 3.13, standard library only (`json`, `math`, `datetime`),
no network, fully deterministic. Run it the way the grader does:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` per case: `{"trophies": [{"id", "q": [q05, q25, q50, q75, q95]}]}`,
55 rows, quantiles in percent, non-decreasing, 4 decimals.

## Model (summary)

Per trophy, on the logit scale against log-time since launch
(19 Nov 2026 00:00Z):

- a weighted least-squares slope from the reads that contain the trophy
  (recent reads count more);
- a read that follows a sharp jump in `players_tracked` and shows a rate
  drop is a refresh dilution, not real decay — it is dropped from the fit,
  so the forecast never chases a denominator artefact down;
- the slope is shrunk toward a prior slope for the trophy's kind, and the
  level is pulled toward a prior adoption curve for that kind; the pull
  grows with the horizon (negligible at 90 minutes, dominant over 4.5 days);
- kinds: platinum and 100% Completion are near-impossible in a week
  (median capped ~0.1–0.2%), story chapters rise fastest (chapter 1 first),
  grinds (250 Stunts, Shooting Range gold, card sets, …) rise slowest;
- the spread is honest: base + `sqrt(horizon)` growth + a term for how
  unidentified the slope is (a frozen history says almost nothing), so the
  90% band is wide exactly when it should be.

Edge cases: a trophy at 100% stays pinned near 99.9%; rates of 0% / 100%
are clamped before the logit; missing (`null`) rates are skipped; a
one-read or frozen history falls back to the kind prior with a wide band
and never crashes.
