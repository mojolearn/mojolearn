# mojolearn benchmark board

Generated 2026-10-06T07:00:52Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | amd / hip |
| GPU | AMD Instinct Mi325X VF |
| GPU driver | 6.12.12 |
| CPU | AMD EPYC 9575F 64-Core Processor (20 logical cores) |
| memory bytes | 168790048768 |
| OS | Ubuntu 24.04.2 LTS |
| Python | 3.12.3 CPython |
| mojolearn | None (wheel None, sha256 None) |
| script commit | - |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, torch 2.6.0+rocm6.4.1.git1ded221d, umap-learn 0.5.12, pynndescent 0.6.0, numba 0.67.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.5.3 |

## How to read this board

- Times are wall milliseconds of the public fit call (trees) or the lane's timed call (classical), median of the timed rounds; min..max beside it.
- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; `ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 the higher one. A ratio is shown only when both arms completed every round in this run, and only against an opponent: our two modes are never divided by each other here.
- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 NumPy function per lane for classical.
- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm whose clock excludes an upload or a fit that ours includes).
- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker protocol as classical; every lane's parameters, rows, timed span and each unavoidable mismatch with its reason are in the cells' `settings.lane_config`. Quality is one float64 NumPy function per lane over each arm's saved outputs.
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU is never raced or reported: the board races only our GPU, against GPU opponents; a race keeps CPU opponents only when it has no GPU opponent (Andrew, Oct 2 2026). A cell of ours on the CPU in an old record is dropped before rendering.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 84 planned, 6 done, 10 failed, 0 unsupported, 68 pending. Cells: 18 (REFUSED 9, ok 9).

Inference cells: 26 (UNKNOWN 18, ok 8).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| classical | kmeans | istella | inertia (lower is better) | - | - | torch-gpu 5.991e+17 |
| classical | kmeans | istella | n_iter | - | - | torch-gpu 91 |
| classical | kmeans | taxi | inertia (lower is better) | - | - | torch-gpu 3.06e+08 |
| classical | kmeans | taxi | n_iter | - | - | torch-gpu 54 |
| classical | knn | taxi | recall_at_k (higher is better) | - | - | torch-gpu 0.999730 |
| classical | knn | taxi | rows_with_repeated_ids | - | - | torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | - | torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | - | torch-gpu nan; torch-gpu-eigh 0.768589 |
| classical | ols | taxi | r2 (higher is better) | - | - | torch-gpu 0.908840; torch-gpu-eigh 0.908822 |
| classical | ols | taxi | rmse (lower is better) | - | - | torch-gpu 4.696376; torch-gpu-eigh 4.696849 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | - | torch-gpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | - | torch-gpu 0.999997 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| classical | kmeans | istella | Xq | - | - | - | - | torch-gpu 0.6 ms (IDENTICAL/arm -) |
| classical | kmeans | taxi | Xq | - | - | - | - | torch-gpu 0.5 ms (IDENTICAL/arm -) |
| classical | ols | istella | Xq | - | - | - | - | torch-gpu 0.5 ms (IDENTICAL/arm -); torch-gpu-eigh 0.5 ms (IDENTICAL/arm -) |
| classical | ols | taxi | Xq | - | - | - | - | torch-gpu 0.5 ms (IDENTICAL/arm -); torch-gpu-eigh 0.5 ms (IDENTICAL/arm -) |
| classical | pca | istella | Xq | - | - | - | - | torch-gpu 0.6 ms (IDENTICAL/arm -) |
| classical | pca | taxi | Xq | - | - | - | - | torch-gpu 0.3 ms (IDENTICAL/arm -) |
| trees | gbdt-categorical | taxi | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-categorical | taxi | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | istella | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | istella | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | taxi | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | taxi | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | istella | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | istella | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | istella | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | istella | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | taxi | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | taxi | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-rank-pairlogit | istella | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-rank-pairlogit | istella | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-rank-yetirank | istella | test | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |
| trees | gbdt-rank-yetirank | istella | large | - | - | - | - | xgboost-gpu - ms (IDENTICAL/arm -) |

## Trees

### gbdt-categorical / taxi (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-categorical.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-depthwise / istella (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-depthwise.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-depthwise / taxi (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-depthwise.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-lossguide / istella (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-lossguide.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-lossguide / taxi (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-lossguide.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-multiclass / istella (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-multiclass.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-multiclass / taxi (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-multiclass.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-rank-pairlogit / istella (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-rank-pairlogit.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: the board's own settings (no NVIDIA harness entry)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

### gbdt-rank-yetirank / istella (rows full, shape -)

race: failed, driver rc 1, log `raw/trees/gbdt-rank-yetirank.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (measured this run) |

config: the board's own settings (no NVIDIA harness entry)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | test | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |
| xgboost-gpu | large | - | - | - | 0 | - | - | - | - | UNKNOWN | UNKNOWN(no inference lines) |

## Classical

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 3406.3 | 3406.3..3406.3 | 1 | - | - | 5132.4 | 3460.8 | inertia=5.991e+17, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| init | "k-means++" |
| max_iter | 300 |
| metric | "euclidean" |
| n_clusters | 8 |
| n_init | 1 |
| seed | 7 |
| tol | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | Xq | 500000 | 0.6 | 0.6..0.6 | 1 | - | - | eval_inertia=1.402e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 538.0 | 538.0..538.0 | 1 | - | - | 3194.4 | 459.9 | inertia=3.06e+08, n_iter=54 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| init | "k-means++" |
| max_iter | 300 |
| metric | "euclidean" |
| n_clusters | 8 |
| n_init | 1 |
| seed | 7 |
| tol | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | - | - | eval_inertia=4.572e+07, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 27.2 | 27.2..27.2 | 1 | - | - | 2866.0 | 3242.7 | recall_at_k=0.999730, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| algorithm | "brute" |
| metric | "euclidean" |
| n_neighbors | 64 |
| p | 2 |
| seed | 7 |

### ols / istella (rows full, shape 2043304x220)

race: failed, driver rc 0, log `logs/classical.ols.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 2340.8 | 2340.8..2340.8 | 1 | - | - | 5648.2 | 5303.6 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 42.8 | 42.8..42.8 | 1 | - | - | 5051.5 | 3649.4 | finite=True, r2=0.151604, rmse=0.768589 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu | torch-gpu-eigh |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| fit_intercept | true | true |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | - | - | predict_max_rel_err_own_fp64=nan, r2_eval=nan, rmse_eval=nan | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu-eigh | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | - | - | predict_max_rel_err_own_fp64=4.831e-07, r2_eval=0.151604, rmse_eval=0.768589 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 100.7 | 100.7..100.7 | 1 | - | - | 3535.9 | 695.3 | finite=True, r2=0.908840, rmse=4.696376 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 34.0 | 34.0..34.0 | 1 | - | - | 3018.3 | 572.0 | finite=True, r2=0.908822, rmse=4.696849 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu | torch-gpu-eigh |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| fit_intercept | true | true |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | - | - | predict_max_rel_err_own_fp64=5.913e-08, r2_eval=0.908840, rmse_eval=4.696376 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu-eigh | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | - | - | predict_max_rel_err_own_fp64=8.007e-08, r2_eval=0.908822, rmse_eval=4.696849 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 19.9 | 19.9..19.9 | 1 | - | - | 5044.4 | 3505.8 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| n_components | 10 |
| seed | 7 |
| svd_solver | "covariance_eigh" |
| whiten | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | Xq | 500000 | 0.6 | 0.6..0.6 | 1 | - | - | transform_max_rel_err_own_fp64=3.575e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 10.9 | 10.9..10.9 | 1 | - | - | 3009.5 | 412.0 | explained_variance_ratio_sum=0.999997 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| n_components | 10 |
| seed | 7 |
| svd_solver | "covariance_eigh" |
| whiten | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | Xq | 500000 | 0.3 | 0.3..0.3 | 1 | - | - | transform_max_rel_err_own_fp64=9.597e-08 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML and cuVS: CUDA only; no ROCm build is pinned.
- Classical, wave 2, not planned on this vendor: faiss-gpu: the pinned FAISS GPU builds are CUDA; faiss-cpu is the arm on this box.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU: never raced or reported; the board races only our GPU (Andrew, Oct 2 2026). The host column gives same-bits digests only (lq ID).
- Neural, not planned: lm-host-train-step, lm-infer, mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: ours runs the CPU binding, and our CPU is never raced, in no numeric mode (the Apple FAST neural tier is the GPU lanes only).
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba opponents are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels, which the board does not install; a Mamba ratio here is against a reference implementation, not a deployment kernel.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on ROCm accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on ROCm has no int8 matmul, so ours races alone

