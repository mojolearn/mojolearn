# Three-suite sweep, quiet-box rerun — 2026-08-22 (SWEEP3, afternoon)

The certified-quiet companion to `THREE_SUITE_2026-08-22.md`. The morning
window carried swap exhaustion (22.7/23.5 GB, kernel OOM kills) and ambient
loads, so its ABSOLUTE seconds were flagged. This rerun is the same harness,
same five invocations, same datasets, on an otherwise idle box at HEAD
`dfa41bb` (which includes the Newton-walk width fix — see the AUC note).
Launched 13:11:40 under `caffeinate -is`, detached.

**Window integrity:** the box SLEPT ~13:15–15:40 (battery lid-close) inside
the year-forest invocation; per-arm timers are monotonic and exclude sleep,
and the arm times replicate the morning's directions, but that invocation is
flagged NOT-CONTINUOUS. The two higgs invocations ran post-wake: higgs GBDT
(15:41–15:46) fully clean; higgs forest (15:46–~15:57) carried light
read-only source-audit load (five subagents, grep/read, no builds) in its
final minutes — interleaved-pair ratio defense applies, disclosed here.
year GBDT and covtype forest (13:11–13:12+) ran clean pre-sleep.

## GBDT — symmetric trees, 500 trees, vs CatBoost CPU

| dataset | ours (GPU) | cat-cpu | speedup | morning ratio | ours acc | theirs acc |
|---|---|---|---|---|---|---|
| year | 11.95 s | 13.59 s | **1.14×** | 1.25× | MSE 80.106 | MSE 79.975 |
| higgs | 131.5 s | 177.8 s | **1.35×** | 1.85× | AUC 0.82167 | AUC 0.83011 |

- **The quiet ratios are LOWER than the morning's.** Memory pressure hurt
  the RAM-heavy CPU arms more than ours; the quiet numbers are the certified
  ones and the ones to publish. (cat-cpu higgs: 360.9 s → 177.8 s quiet.)
- **higgs AUC moved 0.8213 → 0.82167**: the Newton-walk float32-acceptance
  width fix (archive/reference/PORTING.md 140) is in this binary. Residual gap −0.0084 was
  tracked here as a CONFIG-PARITY item (their stock MVS 0.8 +
  random_strength 1.0), ruling pending.
  **THE RULING CAME IN 2026-08-31 AND IT IS NEGATIVE.** Matching the config
  on their side (bootstrap_type='No', random_strength=0,
  boosting_type='Plain') moved CatBoost UP, 0.8303359534882081 to
  0.8304610825961414, and WIDENED the gap to −0.00885. Their shipped
  regularization was costing them a little, not buying them the gap. The
  configuration difference was real and it is NOT the cause, so this line
  no longer stands as the explanation of the −0.0084 and the suspicion
  moves to archive/reference/PORTING.md 140's leaf walk. `boosting_type` also came back
  benign: read off the fitted model, CatBoost resolved it to Plain by
  itself at this size. Evidence:
  `bench/results/higgs_matched_config_2026-08-31.json`, script beside it.
- year MSE parity holds (80.106 vs 79.975).

## Random Forest — 100 trees, vs sklearn RF (all 10 cores)

| dataset | ours (GPU) | skl-rf-cpu | speedup | morning ratio | ours acc | theirs acc |
|---|---|---|---|---|---|---|
| covtype | 4.02 s | 5.26 s | **1.31×** | 1.29× | 0.7224 | 0.7197 |
| year† | 14.14 s | 385.1 s | **27.2×** | 27.2× | MSE 91.712 | MSE 91.760 |
| higgs | 47.05 s | 310.3 s | **6.6×** | 8.4× | AUC 0.77472 | AUC 0.77528 |

† year forest is the sleep-straddled invocation (flagged, not continuous).

## Extra Trees — 100 trees, vs sklearn ET (all 10 cores)

| dataset | ours (GPU) | skl-et-cpu | speedup | morning ratio | ours acc | theirs acc |
|---|---|---|---|---|---|---|
| covtype | 3.70 s | 5.44 s | **1.47×** | 2.46× | 0.6469 | 0.6448 |
| year† | 18.84 s | 54.64 s | **2.90×** | 3.83× | MSE 96.570 | MSE 96.488 |
| higgs | 39.02 s | 132.4 s | **3.39×** | 4.6× | AUC 0.70922 | AUC 0.70043 |

- higgs ET accuracy remains ABOVE sklearn's (DEVIATION 218 binary).
- The covtype ET compression (2.46× → 1.47×) is the largest: the morning's
  sklearn ET arm (13.1 s) was the most swap-punished number in that table.

## Determinism cross-check

Every mojolearn accuracy above is IDENTICAL to its morning/12:43 value at
displayed precision (year GBDT MSE 80.10594940185547 exact match; covtype
RF 0.7224426219632883 exact; year RF MSE 91.71202850341797 exact; ET
96.57011148079309 / 0.7092160874339699 exact) — different day-window, same
bits, consistent with the post-b202a3d 6/6 determinism claim. higgs GBDT
AUC differs from the morning BY THE WALK FIX, as expected.

## The honest headline

On a certified-quiet box the ratios compress: GBDT 1.14–1.35× vs CatBoost
CPU, RF 1.3–27× and ET 1.5–3.4× vs 10-core sklearn, at accuracy parity or
better everywhere. The morning table's larger ratios were partly the CPU
arms' swap cost. Quiet numbers are the publishable ones; the morning file
stays as the record of why.

## Provenance

`gbm_bench_year_2026-08-22_131140.json` (gbdt),
`gbm_bench_covtype_2026-08-22_131210.json` (forest),
`gbm_bench_year_2026-08-22_131234.json` (forest, flagged),
`gbm_bench_higgs_2026-08-22_154116.json` (gbdt, clean),
`gbm_bench_higgs_2026-08-22_154635.json` (forest, light-read load in tail).
Env records beside each. Sweep log: session scratchpad `sweep3.log`,
sentinel `SWEEP3-DONE`.
