# mojolearn benchmark board

Generated 2026-10-06T07:08:42Z from `board.json` (schema `mojolearn-bench-board/1`).

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

Races: 84 planned, 31 done, 11 failed, 0 unsupported, 42 pending. Cells: 77 (REFUSED 9, ok 68).

Inference cells: 72 (UNKNOWN 18, ok 54).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | cholesky | synthetic | relative_residual | - | - | torch-gpu 1.044e-07 |
| algos | eigh | synthetic | max_eigenvalue_error | - | - | torch-gpu 1.254e-06 |
| algos | eigh | synthetic | relative_residual | - | - | torch-gpu 1.269e-06 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | - | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-bf16 0.971788; torch-compile-bf16 0.971788 |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | - | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-bf16 0.865777; torch-compile-bf16 0.865777 |
| algos | gru-reg | synthetic | r2 (higher is better) | - | - | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-bf16 0.981940; torch-compile-bf16 0.981940 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | - | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-bf16 0.155700; torch-compile-bf16 0.155700 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | - | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-bf16 0.748254; torch-compile-bf16 0.748254 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | - | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-bf16 0.544144; torch-compile-bf16 0.544144 |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | - | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | - | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-bf16 0.868056; torch-compile-bf16 0.868056 |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | - | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-bf16 0.981004; torch-compile-bf16 0.981004 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | - | torch-eager-fp32 0.159641; torch-compile-fp32 0.159641; torch-eager-bf16 0.159679; torch-compile-bf16 0.159679 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | - | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-bf16 0.751591; torch-compile-bf16 0.751591 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | - | torch-eager-fp32 0.540429; torch-compile-fp32 0.540429; torch-eager-bf16 0.540526; torch-compile-bf16 0.540526 |
| algos | lstsq | istella | relative_residual | - | - | torch-gpu nan |
| algos | lstsq | taxi | relative_residual | - | - | torch-gpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | - | torch-gpu 4.003e-07 |
| algos | lu-solve | synthetic | relative_residual | - | - | torch-gpu 4.041e-07 |
| algos | qr | istella | relative_gram_difference | - | - | torch-gpu 0.0001698 |
| algos | qr | taxi | relative_gram_difference | - | - | torch-gpu 6.794e-07 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | - | torch-gpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | - | torch-gpu 0.027197 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | - | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-bf16 0.953559; torch-compile-bf16 0.953559 |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | - | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-bf16 0.867947; torch-compile-bf16 0.867947 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | - | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-bf16 0.739017; torch-compile-bf16 0.739017 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | - | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-bf16 0.554037; torch-compile-bf16 0.554037 |
| algos | svd | istella | max_rel_singular_value_error | - | - | torch-gpu 2.707e+08 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | - | torch-gpu 0.026534 |
| algos | svd | taxi | max_rel_singular_value_error | - | - | torch-gpu 4.401e-05 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | - | torch-gpu 0.003043 |
| algos | svgp | istella | r2 (higher is better) | - | - | gpytorch-gpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | - | gpytorch-gpu 0.878383 |
| algos | svgp | taxi | r2 (higher is better) | - | - | gpytorch-gpu -0.209454 |
| algos | svgp | taxi | rmse (lower is better) | - | - | gpytorch-gpu 17.830481 |
| classical | kmeans | istella | inertia (lower is better) | - | - | torch-gpu 5.991e+17 |
| classical | kmeans | istella | n_iter | - | - | torch-gpu 91 |
| classical | kmeans | taxi | inertia (lower is better) | - | - | torch-gpu 3.06e+08 |
| classical | kmeans | taxi | n_iter | - | - | torch-gpu 54 |
| classical | knn | istella | recall_at_k (higher is better) | - | - | torch-gpu 0.978680 |
| classical | knn | istella | rows_with_repeated_ids | - | - | torch-gpu 0 |
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
| algos | gru-clf | synthetic | Xq | - | - | - | - | torch-eager-fp32 6.2 ms (IDENTICAL/arm -); torch-compile-fp32 6.1 ms (IDENTICAL/arm -); torch-eager-bf16 6.7 ms (IDENTICAL/arm -); torch-compile-bf16 6.8 ms (IDENTICAL/arm -) |
| algos | gru-clf | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 6.3 ms (IDENTICAL/arm -); torch-compile-fp32 6.3 ms (IDENTICAL/arm -); torch-eager-bf16 5.9 ms (IDENTICAL/arm -); torch-compile-bf16 6.9 ms (IDENTICAL/arm -) |
| algos | gru-reg | synthetic | Xq | - | - | - | - | torch-eager-fp32 6.1 ms (IDENTICAL/arm -); torch-compile-fp32 6.1 ms (IDENTICAL/arm -); torch-eager-bf16 6.6 ms (IDENTICAL/arm -); torch-compile-bf16 6.8 ms (IDENTICAL/arm -) |
| algos | gru-reg | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 6.1 ms (IDENTICAL/arm -); torch-compile-fp32 6.2 ms (IDENTICAL/arm -); torch-eager-bf16 6.7 ms (IDENTICAL/arm -); torch-compile-bf16 6.7 ms (IDENTICAL/arm -) |
| algos | lstm-clf | synthetic | Xq | - | - | - | - | torch-eager-fp32 3.4 ms (IDENTICAL/arm -); torch-compile-fp32 3.4 ms (IDENTICAL/arm -); torch-eager-bf16 2.9 ms (IDENTICAL/arm -); torch-compile-bf16 3.2 ms (IDENTICAL/arm -) |
| algos | lstm-clf | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 3.1 ms (IDENTICAL/arm -); torch-compile-fp32 3.1 ms (IDENTICAL/arm -); torch-eager-bf16 3.2 ms (IDENTICAL/arm -); torch-compile-bf16 3.2 ms (IDENTICAL/arm -) |
| algos | lstm-reg | synthetic | Xq | - | - | - | - | torch-eager-fp32 3.5 ms (IDENTICAL/arm -); torch-compile-fp32 3.5 ms (IDENTICAL/arm -); torch-eager-bf16 3.3 ms (IDENTICAL/arm -); torch-compile-bf16 3.4 ms (IDENTICAL/arm -) |
| algos | lstm-reg | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 3.0 ms (IDENTICAL/arm -); torch-compile-fp32 3.5 ms (IDENTICAL/arm -); torch-eager-bf16 2.9 ms (IDENTICAL/arm -); torch-compile-bf16 3.4 ms (IDENTICAL/arm -) |
| algos | rnn-clf | synthetic | Xq | - | - | - | - | torch-eager-fp32 2.8 ms (IDENTICAL/arm -); torch-compile-fp32 2.8 ms (IDENTICAL/arm -); torch-eager-bf16 3.0 ms (IDENTICAL/arm -); torch-compile-bf16 3.3 ms (IDENTICAL/arm -) |
| algos | rnn-clf | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 2.2 ms (IDENTICAL/arm -); torch-compile-fp32 2.8 ms (IDENTICAL/arm -); torch-eager-bf16 3.2 ms (IDENTICAL/arm -); torch-compile-bf16 3.3 ms (IDENTICAL/arm -) |
| algos | rnn-reg | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 2.7 ms (IDENTICAL/arm -); torch-compile-fp32 2.4 ms (IDENTICAL/arm -); torch-eager-bf16 2.8 ms (IDENTICAL/arm -); torch-compile-bf16 2.8 ms (IDENTICAL/arm -) |
| algos | svgp | istella | Xq | - | - | - | - | gpytorch-gpu 11.7 ms (IDENTICAL/arm -) |
| algos | svgp | taxi | Xq | - | - | - | - | gpytorch-gpu 11.5 ms (IDENTICAL/arm -) |
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

### knn / istella (rows full, shape 400000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 32.3 | 32.3..32.3 | 1 | - | - | 3187.9 | 3884.6 | recall_at_k=0.978680, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

## Algorithm expansion

### cholesky / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cholesky.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 20.1 | 20.1..20.1 | 1 | - | - | 3745.3 | 768.0 | relative_residual=1.044e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### eigh / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.eigh.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 1922.1 | 1922.1..1922.1 | 1 | - | - | 2900.1 | 1220.6 | max_eigenvalue_error=1.254e-06, relative_residual=1.269e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1946.7 | 1946.7..1946.7 | 1 | - | - | 3217.3 | 324.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1997.8 | 1997.8..1997.8 | 1 | - | - | 3248.0 | 324.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1852.7 | 1852.7..1852.7 | 1 | - | - | 4957.7 | 168.7 | accuracy=0.971788 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1973.8 | 1973.8..1973.8 | 1 | - | - | 5008.5 | 168.7 | accuracy=0.971788 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 6.8 | 6.8..6.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1792.8 | 1792.8..1792.8 | 1 | - | - | 3610.7 | 324.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2027.5 | 2027.5..2027.5 | 1 | - | - | 3258.5 | 324.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2025.2 | 2025.2..2025.2 | 1 | - | - | 5294.2 | 168.7 | accuracy=0.865777 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1899.7 | 1899.7..1899.7 | 1 | - | - | 5008.7 | 168.7 | accuracy=0.865777 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 6.3 | 6.3..6.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 6.3 | 6.3..6.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 5.9 | 5.9..5.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 6.9 | 6.9..6.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1993.7 | 1993.7..1993.7 | 1 | - | - | 3044.7 | 324.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1996.8 | 1996.8..1996.8 | 1 | - | - | 3072.8 | 324.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2078.2 | 2078.2..2078.2 | 1 | - | - | 3656.3 | 168.4 | finite=True, r2=0.981940, rmse=0.155700 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2123.0 | 2123.0..2123.0 | 1 | - | - | 3708.5 | 168.4 | finite=True, r2=0.981940, rmse=0.155700 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 6.6 | 6.6..6.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 6.8 | 6.8..6.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 2012.1 | 2012.1..2012.1 | 1 | - | - | 3044.6 | 324.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2006.3 | 2006.3..2006.3 | 1 | - | - | 3072.3 | 324.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1801.3 | 1801.3..1801.3 | 1 | - | - | 3655.7 | 168.4 | finite=True, r2=0.748254, rmse=0.544144 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2127.7 | 2127.7..2127.7 | 1 | - | - | 3708.4 | 168.4 | finite=True, r2=0.748254, rmse=0.544144 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 6.2 | 6.2..6.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 903.2 | 903.2..903.2 | 1 | - | - | 3208.6 | 350.8 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 894.9 | 894.9..894.9 | 1 | - | - | 3257.9 | 350.8 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 854.9 | 854.9..854.9 | 1 | - | - | 4967.1 | 182.8 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1044.0 | 1044.0..1044.0 | 1 | - | - | 5000.0 | 182.8 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 2.9 | 2.9..2.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 861.1 | 861.1..861.1 | 1 | - | - | 3603.8 | 350.8 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 815.7 | 815.7..815.7 | 1 | - | - | 3258.3 | 350.8 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1039.6 | 1039.6..1039.6 | 1 | - | - | 5292.2 | 182.8 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 977.6 | 977.6..977.6 | 1 | - | - | 5008.3 | 182.8 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 923.1 | 923.1..923.1 | 1 | - | - | 3036.9 | 350.5 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 973.6 | 973.6..973.6 | 1 | - | - | 3082.7 | 350.5 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1053.4 | 1053.4..1053.4 | 1 | - | - | 3660.4 | 182.5 | finite=True, r2=0.981004, rmse=0.159679 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1065.8 | 1065.8..1065.8 | 1 | - | - | 3706.1 | 182.5 | finite=True, r2=0.981004, rmse=0.159679 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 920.3 | 920.3..920.3 | 1 | - | - | 3043.3 | 350.5 | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 976.5 | 976.5..976.5 | 1 | - | - | 3084.4 | 350.5 | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1017.5 | 1017.5..1017.5 | 1 | - | - | 3667.0 | 182.5 | finite=True, r2=0.751591, rmse=0.540526 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1057.7 | 1057.7..1057.7 | 1 | - | - | 3707.8 | 182.5 | finite=True, r2=0.751591, rmse=0.540526 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 3.0 | 3.0..3.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 2.9 | 2.9..2.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 0, log `logs/algos.lstsq.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 1138.9 | 1138.9..1138.9 | 1 | - | - | 4308.1 | 1691.4 | relative_residual=nan | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lstsq.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 33.3 | 33.3..33.3 | 1 | - | - | 3132.5 | 95.4 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-factor.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 86.9 | 86.9..86.9 | 1 | - | - | 2673.1 | 710.0 | relative_residual=4.003e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-solve.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 86.3 | 86.3..86.3 | 1 | - | - | 2694.0 | 712.0 | relative_residual=4.041e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 1852.3 | 1852.3..1852.3 | 1 | - | - | 3625.8 | 2520.6 | relative_gram_difference=0.0001698 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 24.6 | 24.6..24.6 | 1 | - | - | 2690.6 | 126.0 | relative_gram_difference=6.794e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.randomized-svd.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 352.0 | 352.0..352.0 | 1 | - | - | 3836.8 | 1017.6 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| n_components | 8 |
| n_iter | 4 |
| seed | 7 |

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.randomized-svd.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 215.0 | 215.0..215.0 | 1 | - | - | 3037.1 | 228.0 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| n_components | 8 |
| n_iter | 4 |
| seed | 7 |

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 889.2 | 889.2..889.2 | 1 | - | - | 3215.3 | 110.3 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 879.7 | 879.7..879.7 | 1 | - | - | 3255.8 | 110.3 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 950.8 | 950.8..950.8 | 1 | - | - | 4965.1 | 98.7 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 995.6 | 995.6..995.6 | 1 | - | - | 5005.9 | 98.7 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 3.0 | 3.0..3.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 743.5 | 743.5..743.5 | 1 | - | - | 3478.2 | 110.3 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 885.8 | 885.8..885.8 | 1 | - | - | 3256.1 | 110.3 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 975.1 | 975.1..975.1 | 1 | - | - | 5234.9 | 98.7 | accuracy=0.867947 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1002.0 | 1002.0..1002.0 | 1 | - | - | 5006.2 | 98.7 | accuracy=0.867947 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-reg.taxi-hourly.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 863.9 | 863.9..863.9 | 1 | - | - | 3041.2 | 109.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 762.7 | 762.7..762.7 | 1 | - | - | 3082.6 | 109.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1010.8 | 1010.8..1010.8 | 1 | - | - | 3664.6 | 98.4 | finite=True, r2=0.739017, rmse=0.554037 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1033.9 | 1033.9..1033.9 | 1 | - | - | 3705.5 | 98.4 | finite=True, r2=0.739017, rmse=0.554037 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 2.7 | 2.7..2.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 2159.2 | 2159.2..2159.2 | 1 | - | - | 3796.1 | 3361.7 | max_rel_singular_value_error=2.707e+08, relative_reconstruction_error_100k_rows=0.026534 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 271.6 | 271.6..271.6 | 1 | - | - | 2750.5 | 168.0 | max_rel_singular_value_error=4.401e-05, relative_reconstruction_error_100k_rows=0.003043 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | torch-gpu |
|---|---|
| library (source) | torch (declared) |
| seed | "none (deterministic)" |

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpytorch-gpu | gpytorch | gpu | opponent | 48.7 | 48.7..48.7 | 1 | - | - | 4591.1 | 1611.8 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `gpytorch-gpu`, seed 7): MATCHED

| parameter | gpytorch-gpu |
|---|---|
| library (source) | gpytorch (declared) |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpytorch-gpu | Xq | - | 11.7 | 11.7..11.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, gpytorch-gpu: predict(Xq)(Xq)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpytorch-gpu | gpytorch | gpu | opponent | 46.8 | 46.8..46.8 | 1 | - | - | 3618.6 | 1514.9 | finite=True, r2=-0.209454, rmse=17.830481 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `gpytorch-gpu`, seed 7): MATCHED

| parameter | gpytorch-gpu |
|---|---|
| library (source) | gpytorch (declared) |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpytorch-gpu | Xq | - | 11.5 | 11.5..11.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, gpytorch-gpu: predict(Xq)(Xq)

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

