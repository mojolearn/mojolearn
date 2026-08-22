# Three-suite sweep — 2026-08-22 (midday window)

One sweep, five invocations, one process per invocation with all arms
interleaved (the drift defense). NVIDIA's gbm-bench harness, commit
`73a976b0`, datasets at `~/datasets/gbm-bench` (uchg-protected). Box:
M4 Mac16,12, 10 cores, 16GB, AC power, no thermal warning recorded.

**Pairings (Andrew's standing orders):** the symmetric-trees pair is
CatBoost ONLY (LightGBM has no symmetric mode); LightGBM is excluded
from ALL Mac pairs (its arms exist for the NVIDIA leg only); the forest
comparator is multicore sklearn (`n_jobs` = all 10 cores). Same config
both arms of every pair; the device is the only variable.

**Determinism status:** every mojolearn number below is post-b202a3d
(the predrawn-lifetime race sealed, 6/6 bit-identical at 8.8M scale
including contended runs). Accuracy columns are guaranteed
reproducible, not sampled luck.

## GBDT — symmetric trees, 500 trees, vs CatBoost CPU

| dataset | rows | ours (GPU) | cat-cpu | speedup | ours acc | theirs acc |
|---|---|---|---|---|---|---|
| year | 463k | 21.3 s | 26.5 s | **1.25×** | MSE 80.11 | MSE 79.98 |
| higgs | 8.8M | 195.1 s | 360.9 s | **1.85×** | AUC 0.8213 | AUC 0.8301 |

- year MSE parity holds (80.11 vs 79.98; morning window had 1.31–1.68×
  over 3 repeats — this single repeat at 1.25× is in family, low end).
- higgs AUC gap (0.8213 vs 0.8301, −0.009) is the tracked Newton-walk
  divergence (PORTING.md item 140), not noise. It is real and owed.
- The harness Log_Loss column is asymmetric for CatBoost (scores its
  raw output, 1.80 vs our calibrated 0.55) — do not quote it either
  direction; AUC/Accuracy are the comparable columns.
- covtype GBDT is a named refusal: the gbdt python surface has no
  multiclass yet.

## Random Forest — 100 trees, vs sklearn RF (all 10 cores)

| dataset | rows | ours (GPU) | skl-rf-cpu | speedup | ours acc | theirs acc |
|---|---|---|---|---|---|---|
| covtype | 581k | 8.3 s | 10.8 s | **1.29×** | 0.7224 | 0.7193 |
| year | 463k | 19.5 s | 529.6 s | **27.2×** | MSE 91.71 | MSE 91.83 |
| higgs | 8.8M | 76.0 s | 641.2 s | **8.4×** | AUC 0.7747 | AUC 0.7756 |

- Accuracy parity all three rows (covtype +0.003 us; year MSE −0.12
  us; higgs AUC −0.0009 us — all within the RF seed family).
- The year 27× is regression RF: sklearn's regression trees don't bin
  and go deep; this is the cell where GPU binned building dominates.

## Extra Trees — 100 trees, vs sklearn ET (all 10 cores)

| dataset | rows | ours (GPU) | skl-et-cpu | speedup | ours acc | theirs acc |
|---|---|---|---|---|---|---|
| covtype | 581k | 5.3 s | 13.1 s | **2.46×** | 0.6469 | 0.6418 |
| year | 463k | 34.0 s | 130.3 s | **3.83×** | MSE 96.57 | MSE 96.25 |
| higgs | 8.8M | REFUSED | — | — | — | — |

- higgs ET is a named refusal pending extratrees DEVIATION 218 (the
  row-cap lift, approved, queued behind this sweep's ALL-CLEAR).
- Accuracy parity both rows (covtype +0.005 us; year MSE +0.32 them —
  ET seed family).

## Window integrity

- The lock was held for the whole sweep; Andrew's aa_floor.core job is
  a standing contender and was live for parts of the window. The
  interleave-per-invocation design is the defense: every pair shares
  its invocation's ambient load.
- CONTAMINATION FLAG: a foreign benchmark run (lgbm-et-cpu vs our ET,
  `gbm_bench_year_2026-08-22_114036.json`, not part of this sweep and
  in violation of both the lock hold and the LightGBM Mac exclusion)
  ran ~11:40–11:43, inside this sweep's year-forest invocation
  (11:37–11:49). The year RF/ET rows above carried ~2–3 min of extra
  load; the foreign run's own ET timing (31.7 s) replicates this
  sweep's 34.0 s, so the effect is small, but the year forest rows are
  flagged, not certified. covtype (11:36), year gbdt (11:35), and both
  higgs invocations (11:49+) were clean.
- A second out-of-sweep file (`..._112409.json`, 11:24, ours-vs-skl ET
  on year: 33.6 s vs 141.0 s) predates the sweep and independently
  replicates the year ET row.
- Neither foreign file's author has been identified: the coastguard
  session disclaims both, and the extratrees lane session that owned
  ET runs appears to have ended. Treated as an orphaned run; the
  numbers in it are consistent with the sweep's own.
- Ambient load disclosure (coastguard session, same box): latexmk,
  ruff, a ~8 s pytest suite, and a matplotlib job reading ~20M AIS
  rows ran at points during the hour. Ambient CPU load lands on both
  arms of an interleaved invocation, so ratios stand; absolute
  wall-clock times in this window are not thermal-certified (this box
  pins its GPU governor at minimum clock under sustained heat — see
  the box-drift memo).

## Provenance

Sweep JSONs (env record beside each):
`gbm_bench_year_2026-08-22_113523.json` (gbdt),
`gbm_bench_covtype_2026-08-22_113622.json` (forest),
`gbm_bench_year_2026-08-22_113712.json` (forest),
`gbm_bench_higgs_2026-08-22_114918.json` (gbdt),
`gbm_bench_higgs_2026-08-22_115902.json` (rf).
The higgs RF invocation (11:59–12:11) ran clean: no foreign runs, the
box's only benchmark process.
