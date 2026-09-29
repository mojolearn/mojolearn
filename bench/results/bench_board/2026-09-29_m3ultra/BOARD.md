# mojolearn benchmark board

Generated 2026-09-29T07:52:40Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | apple / metal |
| GPU | Apple M3 Ultra |
| GPU driver | macOS 26.7 |
| CPU | Apple M3 Ultra (28 logical cores) |
| memory bytes | 274877906944 |
| OS | macOS 26.7 |
| Python | 3.13.15 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-macosx_11_0_arm64.whl, sha256 ee1187c950c29e791c906cb3beb1c45206e4d8534fea897bc839db4505b38548) |
| script commit | 4aa3e28c02a197c9f2e4205fc275603af447576f |
| modes | fast, identical |
| rounds | 5 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, torch 2.13.0, umap-learn 0.5.12, pynndescent 0.6.0, numba 0.67.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.5.3 |

## How to read this board

- Times are wall milliseconds of the public fit call (trees) or the lane's timed call (classical), median of the timed rounds; min..max beside it.
- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; `ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 the higher one. A ratio is shown only when both arms completed every round in this run, and only against an opponent: our two modes are never divided by each other here.
- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 NumPy function per lane for classical.
- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm whose clock excludes an upload or a fit that ours includes).
- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker protocol as classical; every lane's parameters, rows, timed span and each unavoidable mismatch with its reason are in the cells' `settings.lane_config`. Quality is one float64 NumPy function per lane over each arm's saved outputs.
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. The `*-infer` lanes are the CPU *Inference classes and race `torch-cpu-*` arms. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU tier (`mojolearn CPU IDENTICAL`, arm `ours-cpu`): the same public estimator in a worker started under MOJOLEARN_VENDOR=cpu, the wheel's CPU switch (no GPU set loads; the host bindings answer, IDENTICAL only), read back as vendor cpu or refused by name. It races in the same rounds as every arm; `ours CPU / arm` is its median over each opponent's. `bits_equal_vs_ours_identical` compares its output with our GPU IDENTICAL arm's, bit for bit. Our CPU and GPU times are never divided by each other here.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 396 planned, 12 done, 0 failed, 384 pending. Cells: 52 (PARTIAL 1, REFUSED 8, ok 43).

Inference cells: 92 (UNKNOWN 6, ok 86).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| trees | et | istella | auc (higher is better) | 0.937987 | 0.937987 | - | sklearn-et-cpu 0.937904; lightgbm-cpu 0.923891; cuml-et-gpu - |
| trees | et | istella | logloss (lower is better) | 0.189989 | 0.189989 | - | sklearn-et-cpu 0.190078; lightgbm-cpu 0.224411; cuml-et-gpu - |
| trees | et | taxi | auc (higher is better) | 0.618907 | 0.618907 | - | sklearn-et-cpu 0.618972; lightgbm-cpu 0.601896; cuml-et-gpu - |
| trees | et | taxi | logloss (lower is better) | 0.526142 | 0.526142 | - | sklearn-et-cpu 0.525976; lightgbm-cpu 0.543395; cuml-et-gpu - |
| trees | gbdt-depthwise | istella | auc (higher is better) | 0.972339 | 0.972354 | - | catboost-cpu 0.971850; xgboost-cpu 0.973663 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | 0.127070 | 0.126973 | - | catboost-cpu 0.128459; xgboost-cpu 0.125560 |
| trees | gbdt-depthwise | taxi | auc (higher is better) | 0.626895 | 0.626771 | - | catboost-cpu 0.625512; xgboost-cpu 0.625131 |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | 0.522901 | 0.522947 | - | catboost-cpu 0.523339; xgboost-cpu 0.523404 |
| trees | gbdt-lossguide | istella | auc (higher is better) | 0.974908 | 0.974734 | - | catboost-cpu 0.971558; xgboost-cpu 0.973663; lightgbm-cpu - |
| trees | gbdt-lossguide | istella | logloss (lower is better) | 0.122714 | 0.123184 | - | catboost-cpu 0.129022; xgboost-cpu 0.125560; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | auc (higher is better) | 0.627156 | 0.627224 | - | catboost-cpu 0.624829; xgboost-cpu 0.625131; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | 0.522782 | 0.522759 | - | catboost-cpu 0.523499; xgboost-cpu 0.523404; lightgbm-cpu - |
| trees | gbdt-symmetric | istella | auc (higher is better) | 0.966627 | 0.966627 | - | catboost-cpu 0.966355 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | 0.139668 | 0.139668 | - | catboost-cpu 0.140274 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | 0.621522 | 0.621613 | - | catboost-cpu 0.620716 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | 0.524425 | 0.524402 | - | catboost-cpu 0.524599 |
| trees | iforest | istella | auc (higher is better) | 0.830358 | 0.830358 | - | sklearn-iforest-cpu 0.827914; cuml-iforest-gpu - |
| trees | iforest | taxi | auc (higher is better) | 0.551846 | 0.551846 | - | sklearn-iforest-cpu 0.552849; cuml-iforest-gpu - |
| trees | rf | istella | auc (higher is better) | 0.965173 | 0.965173 | - | sklearn-rf-cpu 0.964715; lightgbm-cpu 0.947195; cuml-rf-gpu - |
| trees | rf | istella | logloss (lower is better) | 0.144571 | 0.144571 | - | sklearn-rf-cpu 0.145241; lightgbm-cpu 0.201578; cuml-rf-gpu - |
| trees | rf | taxi | auc (higher is better) | 0.625440 | 0.625440 | - | sklearn-rf-cpu 0.625649; lightgbm-cpu 0.616567; cuml-rf-gpu - |
| trees | rf | taxi | logloss (lower is better) | 0.523500 | 0.523500 | - | sklearn-rf-cpu 0.523126; lightgbm-cpu 0.532734; cuml-rf-gpu - |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| trees | et | istella | test | 500000 | 42.3 | 46.3 | yes | - | - | sklearn-et-cpu 246.2 ms (IDENTICAL/arm 0.188); lightgbm-cpu 324.3 ms (IDENTICAL/arm 0.143) |
| trees | et | istella | large | 1000000 | 84.6 | 100.6 | yes | - | - | sklearn-et-cpu 476.4 ms (IDENTICAL/arm 0.211); lightgbm-cpu 646.2 ms (IDENTICAL/arm 0.156) |
| trees | et | taxi | test | 500000 | 16.6 | 21.5 | yes | - | - | sklearn-et-cpu 138.8 ms (IDENTICAL/arm 0.155); lightgbm-cpu 89.5 ms (IDENTICAL/arm 0.241) |
| trees | et | taxi | large | 1000000 | 32.0 | 37.2 | yes | - | - | sklearn-et-cpu 248.8 ms (IDENTICAL/arm 0.150); lightgbm-cpu 178.6 ms (IDENTICAL/arm 0.209) |
| trees | gbdt-depthwise | istella | test | 500000 | 116.5 | 115.5 | no | - | - | catboost-cpu 67.3 ms (IDENTICAL/arm 1.717); xgboost-cpu 22.5 ms (IDENTICAL/arm 5.145) |
| trees | gbdt-depthwise | istella | large | 1000000 | 170.9 | 176.3 | no | - | - | catboost-cpu 129.2 ms (IDENTICAL/arm 1.364); xgboost-cpu 41.7 ms (IDENTICAL/arm 4.229) |
| trees | gbdt-depthwise | taxi | test | 500000 | 69.3 | 72.6 | no | - | - | catboost-cpu 29.7 ms (IDENTICAL/arm 2.446); xgboost-cpu 16.1 ms (IDENTICAL/arm 4.511) |
| trees | gbdt-depthwise | taxi | large | 1000000 | 77.8 | 85.1 | no | - | - | catboost-cpu 54.9 ms (IDENTICAL/arm 1.551); xgboost-cpu 29.4 ms (IDENTICAL/arm 2.892) |
| trees | gbdt-lossguide | istella | test | 500000 | 120.6 | 121.5 | no | - | - | catboost-cpu 67.1 ms (IDENTICAL/arm 1.812); xgboost-cpu 22.2 ms (IDENTICAL/arm 5.470); lightgbm-cpu - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | istella | large | 1000000 | 170.8 | 172.5 | no | - | - | catboost-cpu 127.6 ms (IDENTICAL/arm 1.351); xgboost-cpu 42.0 ms (IDENTICAL/arm 4.112); lightgbm-cpu - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | test | 500000 | 68.3 | 72.5 | no | - | - | catboost-cpu 30.2 ms (IDENTICAL/arm 2.400); xgboost-cpu 16.8 ms (IDENTICAL/arm 4.306); lightgbm-cpu - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | large | 1000000 | 77.6 | 85.2 | no | - | - | catboost-cpu 54.7 ms (IDENTICAL/arm 1.557); xgboost-cpu 29.7 ms (IDENTICAL/arm 2.867); lightgbm-cpu - ms (IDENTICAL/arm -) |
| trees | gbdt-symmetric | istella | test | 500000 | 60.1 | 56.5 | no | - | - | catboost-cpu 40.4 ms (IDENTICAL/arm 1.400) |
| trees | gbdt-symmetric | istella | large | 1000000 | 99.9 | 98.5 | no | - | - | catboost-cpu 75.0 ms (IDENTICAL/arm 1.313) |
| trees | gbdt-symmetric | taxi | test | 500000 | 7.7 | 6.1 | no | - | - | catboost-cpu 10.4 ms (IDENTICAL/arm 0.581) |
| trees | gbdt-symmetric | taxi | large | 1000000 | 13.9 | 11.1 | no | - | - | catboost-cpu 16.2 ms (IDENTICAL/arm 0.686) |
| trees | iforest | istella | test | 500000 | 722.8 | 721.7 | no | - | - | sklearn-iforest-cpu 864.6 ms (IDENTICAL/arm 0.835) |
| trees | iforest | istella | large | 1000000 | 1130.5 | 1134.9 | no | - | - | sklearn-iforest-cpu 1702.5 ms (IDENTICAL/arm 0.667) |
| trees | iforest | taxi | test | 500000 | 135.2 | 136.7 | no | - | - | sklearn-iforest-cpu 836.4 ms (IDENTICAL/arm 0.163) |
| trees | iforest | taxi | large | 1000000 | 173.5 | 175.2 | no | - | - | sklearn-iforest-cpu 1657.4 ms (IDENTICAL/arm 0.106) |
| trees | rf | istella | test | 500000 | 47.0 | 49.8 | yes | - | - | sklearn-rf-cpu 252.8 ms (IDENTICAL/arm 0.197); lightgbm-cpu - ms (IDENTICAL/arm -) |
| trees | rf | istella | large | 1000000 | 93.0 | 97.8 | yes | - | - | sklearn-rf-cpu 508.7 ms (IDENTICAL/arm 0.192); lightgbm-cpu - ms (IDENTICAL/arm -) |
| trees | rf | taxi | test | 500000 | 16.1 | 23.9 | yes | - | - | sklearn-rf-cpu 165.9 ms (IDENTICAL/arm 0.144); lightgbm-cpu 346.9 ms (IDENTICAL/arm 0.069) |
| trees | rf | taxi | large | 1000000 | 31.4 | 39.4 | yes | - | - | sklearn-rf-cpu 344.7 ms (IDENTICAL/arm 0.114); lightgbm-cpu 689.7 ms (IDENTICAL/arm 0.057) |

## Trees

### et / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/et.istella.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4394.2 | 4383.8..4413.0 | 5 | - | - | - | 7683.7 | - | auc=0.937987, logloss=0.189989 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4363.8 | 4360.2..4389.4 | 5 | - | - | - | 8953.8 | - | auc=0.937987, logloss=0.189989 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 30429.6 | 30294.3..30523.6 | 5 | 0.144 | 0.143 | - | 5511.7 | - | auc=0.937904, logloss=0.190078 | no | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 127140.5 | 127009.5..127244.3 | 5 | 0.035 | 0.034 | - | 13652.8 | - | auc=0.923891, logloss=0.224411 | yes | NOT-COMPARABLE | - | ok |
| cuml-et-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(ModuleNotFoundError: No module named 'cuml') |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, cuml-et-gpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:146262,ours:1029236,ours-ab:1029236,sklearn-et-cpu:999969 spread=0.8579 verdict=NOT-COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 46.3 | 45.9..48.2 | 5 | - | - | - | auc=0.937987, auc_matches_fit=True, logloss=0.189989, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 42.3 | 41.9..44.2 | 5 | - | - | - | auc=0.937987, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.189989, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 246.2 | 241.8..246.7 | 5 | 0.188 | 0.172 | - | auc=0.937904, auc_matches_fit=True, logloss=0.190078, logloss_matches_fit=True | no | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 324.3 | 321.5..326.7 | 5 | 0.143 | 0.131 | - | auc=0.923891, auc_matches_fit=True, logloss=0.224411, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 100.6 | 88.1..102.6 | 5 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 84.6 | 84.3..84.8 | 5 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 476.4 | 457.8..544.9 | 5 | 0.211 | 0.177 | - | - | no | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 646.2 | 641.3..650.1 | 5 | 0.156 | 0.131 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### et / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/et.taxi.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3407.4 | 3375.9..3412.9 | 5 | - | - | - | 2651.8 | - | auc=0.618907, logloss=0.526142 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3367.5 | 3357.8..3386.0 | 5 | - | - | - | 2658.9 | - | auc=0.618907, logloss=0.526142 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 20743.5 | 20441.3..20830.9 | 5 | 0.164 | 0.162 | - | 4041.4 | - | auc=0.618972, logloss=0.525976 | no | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 4867.0 | 4857.4..4872.6 | 5 | 0.700 | 0.692 | - | 1655.3 | - | auc=0.601896, logloss=0.543395 | yes | NOT-COMPARABLE | - | ok |
| cuml-et-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(ModuleNotFoundError: No module named 'cuml') |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, cuml-et-gpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:3538,ours:881399,ours-ab:881399,sklearn-et-cpu:916826 spread=0.9961 verdict=NOT-COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 21.5 | 18.2..24.9 | 5 | - | - | - | auc=0.618907, auc_matches_fit=True, logloss=0.526142, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 16.6 | 16.5..18.3 | 5 | - | - | - | auc=0.618907, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.526142, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 138.8 | 133.5..203.7 | 5 | 0.155 | 0.120 | - | auc=0.618972, auc_matches_fit=True, logloss=0.525976, logloss_matches_fit=True | no | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 89.5 | 82.1..96.0 | 5 | 0.241 | 0.186 | - | auc=0.601896, auc_matches_fit=True, logloss=0.543395, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 37.2 | 32.8..43.1 | 5 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 32.0 | 31.7..34.9 | 5 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 248.8 | 226.0..314.4 | 5 | 0.150 | 0.128 | - | - | no | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 178.6 | 174.8..187.5 | 5 | 0.209 | 0.179 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.istella.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4165.3 | 4160.8..4169.2 | 5 | - | - | - | 11683.4 | - | auc=0.972354, logloss=0.126973 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3764.7 | 3752.9..3789.6 | 5 | - | - | - | 11670.5 | - | auc=0.972339, logloss=0.127070 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 15484.3 | 15430.9..15503.6 | 5 | 0.269 | 0.243 | - | 9419.5 | - | auc=0.971850, logloss=0.128459 | yes | COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 4979.5 | 4934.2..4995.4 | 5 | 0.836 | 0.756 | - | 9171.8 | - | auc=0.973663, logloss=0.125560 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6351,ours:6345,ours-ab:6255,xgboost-cpu:6296 spread=0.0151 verdict=COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 115.5 | 115.1..121.3 | 5 | - | - | - | auc=0.972354, auc_matches_fit=True, logloss=0.126973, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 116.5 | 112.4..118.3 | 5 | - | - | - | auc=0.972339, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.127070, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.577393 | yes | COMPARABLE | ok |
| catboost-cpu | test | 500000 | 67.3 | 66.5..68.4 | 5 | 1.717 | 1.732 | - | auc=0.971850, auc_matches_fit=True, logloss=0.128459, logloss_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 22.5 | 21.3..24.8 | 5 | 5.145 | 5.188 | - | auc=0.973663, auc_matches_fit=True, logloss=0.125560, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 176.3 | 174.9..179.6 | 5 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 170.9 | 167.7..176.9 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.454613 | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 129.2 | 127.5..129.6 | 5 | 1.364 | 1.322 | - | - | yes | COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 41.7 | 39.1..47.9 | 5 | 4.229 | 4.099 | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.taxi.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3450.2 | 3446.7..3470.4 | 5 | - | - | - | 4871.5 | - | auc=0.626771, logloss=0.522947 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3110.7 | 3095.6..3120.2 | 5 | - | - | - | 4834.5 | - | auc=0.626895, logloss=0.522901 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 8262.8 | 8243.1..8266.5 | 5 | 0.418 | 0.376 | - | 4328.6 | - | auc=0.625512, logloss=0.523339 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 1862.8 | 1857.0..1871.0 | 5 | 1.852 | 1.670 | - | 4072.6 | - | auc=0.625131, logloss=0.523404 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:5749,ours:5903,ours-ab:5582,xgboost-cpu:3911 spread=0.3375 verdict=NOT-COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 72.6 | 71.2..74.2 | 5 | - | - | - | auc=0.626771, auc_matches_fit=True, logloss=0.522947, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 69.3 | 68.6..70.0 | 5 | - | - | - | auc=0.626895, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.522901, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.223931 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 29.7 | 28.4..30.8 | 5 | 2.446 | 2.333 | - | auc=0.625512, auc_matches_fit=True, logloss=0.523339, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 16.1 | 15.1..17.1 | 5 | 4.511 | 4.304 | - | auc=0.625131, auc_matches_fit=True, logloss=0.523404, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 85.1 | 83.2..87.4 | 5 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 77.8 | 76.8..79.3 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.431454 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 54.9 | 53.7..55.5 | 5 | 1.551 | 1.418 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 29.4 | 26.9..30.7 | 5 | 2.892 | 2.645 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.istella.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7701.1 | 7682.4..7727.6 | 5 | - | - | - | 11698.7 | - | auc=0.974734, logloss=0.123184 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7532.1 | 7505.1..7689.2 | 5 | - | - | - | 11692.5 | - | auc=0.974908, logloss=0.122714 | no | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 18114.3 | 18050.2..18165.8 | 5 | 0.425 | 0.416 | - | 9482.3 | - | auc=0.971558, logloss=0.129022 | yes | COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 7574.7 | 7557.0..7586.9 | 5 | 1.017 | 0.994 | - | 9235.9 | - | auc=0.973663, logloss=0.125560 | yes | COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(LightGBMError during warm-up: Check failed: (best_split_info.left_count) > (0) at /Users/runner/work/LightGBM/LightGBM/l) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6366,ours:6396,ours-ab:6395,xgboost-cpu:6296 spread=0.0156 verdict=COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 121.5 | 119.2..128.4 | 5 | - | - | - | auc=0.974734, auc_matches_fit=True, logloss=0.123184, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 120.6 | 118.0..122.7 | 5 | - | - | - | auc=0.974908, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.122714, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.355339 | yes | COMPARABLE | ok |
| catboost-cpu | test | 500000 | 67.1 | 65.6..67.7 | 5 | 1.812 | 1.798 | - | auc=0.971558, auc_matches_fit=True, logloss=0.129022, logloss_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 22.2 | 21.7..24.1 | 5 | 5.470 | 5.430 | - | auc=0.973663, auc_matches_fit=True, logloss=0.125560, logloss_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cpu | test | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 172.5 | 166.1..175.0 | 5 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 170.8 | 168.1..171.8 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.402757 | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 127.6 | 127.0..136.9 | 5 | 1.351 | 1.338 | - | - | yes | COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 42.0 | 41.1..43.6 | 5 | 4.112 | 4.072 | - | - | yes | COMPARABLE | ok |
| lightgbm-cpu | large | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.taxi.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6586.2 | 6545.7..6600.8 | 5 | - | - | - | 4493.5 | - | auc=0.627224, logloss=0.522759 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6329.9 | 6309.2..6341.2 | 5 | - | - | - | 4486.1 | - | auc=0.627156, logloss=0.522782 | no | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 12084.8 | 12051.7..12109.0 | 5 | 0.545 | 0.524 | - | 4015.1 | - | auc=0.624829, logloss=0.523499 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 3241.4 | 3237.1..3244.1 | 5 | 2.032 | 1.953 | - | 3752.8 | - | auc=0.625131, logloss=0.523404 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: Check failed: (best_split_info.left_count) > (0) at /Users/runner/work/LightGBM/LightGBM/l) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:5743,ours:6239,ours-ab:6224,xgboost-cpu:3911 spread=0.3731 verdict=NOT-COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 72.5 | 71.1..74.8 | 5 | - | - | - | auc=0.627224, auc_matches_fit=True, logloss=0.522759, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 68.3 | 68.0..69.5 | 5 | - | - | - | auc=0.627156, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.522782, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.247510 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 30.2 | 28.4..30.7 | 5 | 2.400 | 2.261 | - | auc=0.624829, auc_matches_fit=True, logloss=0.523499, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 16.8 | 15.6..17.4 | 5 | 4.306 | 4.057 | - | auc=0.625131, auc_matches_fit=True, logloss=0.523404, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 85.2 | 84.7..87.2 | 5 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 77.6 | 77.3..79.0 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.242997 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 54.7 | 53.7..55.2 | 5 | 1.557 | 1.419 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 29.7 | 29.5..32.4 | 5 | 2.867 | 2.613 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.istella.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3123.1 | 3107.7..3127.7 | 5 | - | - | - | 11302.8 | - | auc=0.966627, logloss=0.139668 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2924.2 | 2922.5..2933.0 | 5 | - | - | - | 11281.6 | - | auc=0.966627, logloss=0.139668 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 11376.3 | 11327.6..11414.5 | 5 | 0.275 | 0.257 | - | 9121.6 | - | auc=0.966355, logloss=0.140274 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:6400,ours:6400,ours-ab:6400 spread=0.0000 verdict=COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 56.5 | 52.9..59.0 | 5 | - | - | - | auc=0.966627, auc_matches_fit=True, logloss=0.139668, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 60.1 | 56.7..61.9 | 5 | - | - | - | auc=0.966627, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.139668, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.001486 | yes | COMPARABLE | ok |
| catboost-cpu | test | 500000 | 40.4 | 40.0..43.3 | 5 | 1.400 | 1.488 | - | auc=0.966355, auc_matches_fit=True, logloss=0.140274, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 98.5 | 95.2..100.6 | 5 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 99.9 | 96.3..102.7 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.002330 | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 75.0 | 74.6..78.2 | 5 | 1.313 | 1.332 | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.taxi.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1718.7 | 1717.1..1724.7 | 5 | - | - | - | 4164.6 | - | auc=0.621613, logloss=0.524402 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1473.5 | 1472.8..1478.6 | 5 | - | - | - | 4133.5 | - | auc=0.621522, logloss=0.524425 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 4927.2 | 4913.4..5001.6 | 5 | 0.349 | 0.299 | - | 3822.8 | - | auc=0.620716, logloss=0.524599 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:6352,ours:6400,ours-ab:6400 spread=0.0075 verdict=COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 6.1 | 6.0..6.2 | 5 | - | - | - | auc=0.621613, auc_matches_fit=True, logloss=0.524402, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 7.7 | 7.6..8.0 | 5 | - | - | - | auc=0.621522, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.524425, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.061885 | yes | COMPARABLE | ok |
| catboost-cpu | test | 500000 | 10.4 | 9.2..10.7 | 5 | 0.581 | 0.739 | - | auc=0.620716, auc_matches_fit=True, logloss=0.524599, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 11.1 | 9.6..13.7 | 5 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 13.9 | 11.7..16.8 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.059456 | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 16.2 | 15.2..16.4 | 5 | 0.686 | 0.859 | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### iforest / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/iforest.istella.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 446.7 | 443.7..449.6 | 5 | - | - | - | 7019.5 | - | auc=0.830358 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 446.2 | 441.0..446.6 | 5 | - | - | - | 7151.9 | - | auc=0.830358 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 280.9 | 261.7..285.9 | 5 | 1.590 | 1.588 | - | 4019.5 | - | auc=0.827914 | yes | UNKNOWN | - | ok |
| cuml-iforest-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(ModuleNotFoundError: No module named 'cuml') |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, cuml-iforest-gpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:4534 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 721.7 | 700.8..726.2 | 5 | - | - | - | auc=0.830358, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn FAST | test | 500000 | 722.8 | 720.9..732.0 | 5 | - | - | - | auc=0.830358, auc_matches_fit=True, bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.96e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | test | 500000 | 864.6 | 848.9..868.7 | 5 | 0.835 | 0.836 | - | auc=0.827914, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 1134.9 | 1133.2..1137.0 | 5 | - | - | - | - | yes | UNKNOWN | ok |
| mojolearn FAST | large | 1000000 | 1130.5 | 1130.4..1140.1 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=8.941e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | large | 1000000 | 1702.5 | 1699.0..1711.4 | 5 | 0.667 | 0.664 | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, ours-ab: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, sklearn-iforest-cpu: sklearn IsolationForest.score_samples(X), n_jobs -1

### iforest / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/iforest.taxi.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 121.6 | 120.1..122.2 | 5 | - | - | - | 1413.3 | - | auc=0.551846 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 120.4 | 119.0..120.8 | 5 | - | - | - | 1477.7 | - | auc=0.551846 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 469.7 | 464.9..513.4 | 5 | 0.259 | 0.256 | - | 1279.8 | - | auc=0.552849 | yes | UNKNOWN | - | ok |
| cuml-iforest-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(ModuleNotFoundError: No module named 'cuml') |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, cuml-iforest-gpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:6131 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 136.7 | 132.6..139.4 | 5 | - | - | - | auc=0.551846, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn FAST | test | 500000 | 135.2 | 132.0..137.3 | 5 | - | - | - | auc=0.551846, auc_matches_fit=True, bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.96e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | test | 500000 | 836.4 | 835.2..839.9 | 5 | 0.163 | 0.162 | - | auc=0.552849, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 175.2 | 174.0..177.8 | 5 | - | - | - | - | yes | UNKNOWN | ok |
| mojolearn FAST | large | 1000000 | 173.5 | 172.5..173.7 | 5 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.96e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | large | 1000000 | 1657.4 | 1655.6..1687.3 | 5 | 0.106 | 0.105 | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, ours-ab: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, sklearn-iforest-cpu: sklearn IsolationForest.score_samples(X), n_jobs -1

### rf / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/rf.istella.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6487.9 | 6472.6..6498.8 | 5 | - | - | - | 11534.3 | - | auc=0.965173, logloss=0.144571 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5905.6 | 5881.8..5973.5 | 5 | - | - | - | 11484.4 | - | auc=0.965173, logloss=0.144571 | yes | COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 34126.6 | 33633.7..34244.7 | 5 | 0.190 | 0.173 | - | 10015.5 | - | auc=0.964715, logloss=0.145241 | no | COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 1.131e+06 | 1.13e+06..1.131e+06 | 3 | - | - | - | 17127.7 | - | auc=0.947195, logloss=0.201578 | yes | COMPARABLE | - | PARTIAL(3/5 rounds) |
| cuml-rf-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(ModuleNotFoundError: No module named 'cuml') |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, cuml-rf-gpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:1569514,ours:1550974,ours-ab:1550974,sklearn-rf-cpu:1426068 spread=0.0914 verdict=COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 49.8 | 48.8..53.4 | 5 | - | - | - | auc=0.965173, auc_matches_fit=True, logloss=0.144571, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 47.0 | 46.2..47.1 | 5 | - | - | - | auc=0.965173, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.144571, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | COMPARABLE | ok |
| sklearn-rf-cpu | test | 500000 | 252.8 | 242.2..309.7 | 5 | 0.197 | 0.186 | - | auc=0.964715, auc_matches_fit=True, logloss=0.145241, logloss_matches_fit=True | no | COMPARABLE | ok |
| lightgbm-cpu | test | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 97.8 | 94.7..99.3 | 5 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 93.0 | 92.2..93.2 | 5 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | COMPARABLE | ok |
| sklearn-rf-cpu | large | 1000000 | 508.7 | 484.0..573.1 | 5 | 0.192 | 0.183 | - | - | no | COMPARABLE | ok |
| lightgbm-cpu | large | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, sklearn-rf-cpu: sklearn RandomForestClassifier.predict_proba(X), n_jobs -1, column 1

### rf / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/rf.taxi.rows-full.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5455.7 | 5444.0..5463.9 | 5 | - | - | - | 3194.9 | - | auc=0.625440, logloss=0.523500 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5268.0 | 5262.7..5269.2 | 5 | - | - | - | 3171.2 | - | auc=0.625440, logloss=0.523500 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 24599.3 | 24586.6..24731.8 | 5 | 0.222 | 0.214 | - | 7244.0 | - | auc=0.625649, logloss=0.523126 | no | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 352229.5 | 352061.4..352310.7 | 5 | 0.015 | 0.015 | - | 2523.1 | - | auc=0.616567, logloss=0.532734 | yes | NOT-COMPARABLE | - | ok |
| cuml-rf-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(ModuleNotFoundError: No module named 'cuml') |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, cuml-rf-gpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:463191,ours:1718168,ours-ab:1718168,sklearn-rf-cpu:1260387 spread=0.7304 verdict=NOT-COMPARABLE`

parameters: NOT CHECKED

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 23.9 | 21.3..25.9 | 5 | - | - | - | auc=0.625440, auc_matches_fit=True, logloss=0.523500, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 16.1 | 15.5..16.3 | 5 | - | - | - | auc=0.625440, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.523500, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-rf-cpu | test | 500000 | 165.9 | 162.7..185.0 | 5 | 0.144 | 0.097 | - | auc=0.625649, auc_matches_fit=True, logloss=0.523126, logloss_matches_fit=True | no | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 346.9 | 343.5..361.2 | 5 | 0.069 | 0.046 | - | auc=0.616567, auc_matches_fit=True, logloss=0.532734, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 39.4 | 33.0..43.2 | 5 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 31.4 | 31.4..31.7 | 5 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-rf-cpu | large | 1000000 | 344.7 | 297.6..354.4 | 5 | 0.114 | 0.091 | - | - | no | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 689.7 | 672.5..717.5 | 5 | 0.057 | 0.045 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, sklearn-rf-cpu: sklearn RandomForestClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML and cuVS: CUDA only; no Apple build exists.
- Classical, wave 2, not planned on this vendor: faiss-gpu: CUDA only; faiss-cpu is the arm on this box.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU tier: `--no-cpu-arm` was passed, so no `ours-cpu` arm ran.
- Our CPU tier, no ours-cpu arm: neural mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: its `ours` arm already IS the CPU path (the public *Inference class runs on the host binding).
- Our CPU tier: a GBDT configuration the host side does not restate refuses by name in its ours-cpu cell (python/mojolearn/host_surface.py NO_CPU_PATH lists them), and a FAST-only run (`--modes fast`) has no ours-cpu arm: the host bindings build IDENTICAL only.
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba opponents are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels, which the board does not install; a Mamba ratio here is against a reference implementation, not a deployment kernel.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on MPS accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only

