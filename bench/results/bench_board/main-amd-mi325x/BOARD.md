# mojolearn benchmark board

Generated 2026-10-10T23:05:32Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD amd-mi325x, version label main@ca25d9321. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 153 races; oldest cell main@89485bc96 (2026-10-08T13:27:06Z), newest cell main@ca25d9321 (2026-10-10T19:37:56Z). Boxes: AMD Instinct MI325X (amd, DO).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 176 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: one scored run per cell (lq RACE ALGOS lines, lq CMD bench_board summaries); the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 80, n/a 73).

> Opponents: copied from the stored opponent boards (opponents-20261006, release-board-resume-r2), never re-run here; `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 2 races.

> Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax. A cell of ours whose output hash equals a copied opponent's shows quality identical_to=<arm> (the same bits) where the own-host reference gave none.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 80, n/a 73.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 0 failed runs.


## Box

| field | value |
|---|---|
| vendor / API | amd / hip |
| GPU | AMD Instinct MI325X |
| GPU driver | - |
| CPU | None (None logical cores) |
| memory bytes | - |
| OS | - |
| Python | - |
| mojolearn | main@ca25d9321 (wheel none (unreleased; built from source at each cell's commit), sha256 -) |
| script commit | ca25d932101a8f5bf9f55d5ec5bf26fc1a3909eb |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions |  |

## How to read this board

- Times are wall milliseconds of the public fit call (trees) or the lane's timed call (classical), median of the timed rounds; min..max beside it.
- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; `ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 the higher one. A ratio is shown only when both arms completed every round in this run, and only against an opponent: our two modes are never divided by each other here.
- Two clocks (AGENTS.md measurement item 6): `whole ms` is the operation including the host-to-device copy of its inputs, `kernel ms` the same with the inputs already on the device; `copy ms` comes only from a stored field, named beside it (`upload_ms_separate`: our separate upload probe, kernel = median - copy; `upload_ms_untimed`: an opponent's pre-clock upload, whole = median + copy; `cpu-arm`: no device copy exists). A clock the stored fields cannot give is `-`, never estimated. `ours IDENTICAL / arm (clock)` reads a torch GPU arm on kernel/kernel and every other arm on whole/whole; when that clock is missing on a side it falls back to the other common clock (labelled), and with no common clock it is the two stored medians labelled MIXED with each side's clock.
- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 NumPy function per lane for classical.
- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm whose clock excludes an upload or a fit that ours includes).
- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker protocol as classical; every lane's parameters, rows, timed span and each unavoidable mismatch with its reason are in the cells' `settings.lane_config`. Quality is one float64 NumPy function per lane over each arm's saved outputs.
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU is never raced or reported: the board races only our GPU, against GPU opponents; a race keeps CPU opponents only when it has no GPU opponent (Andrew, Oct 2 2026). A cell of ours on the CPU in an old record is dropped before rendering.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 153 planned, 153 done, 0 failed, 0 unsupported, 0 pending. Cells: 472 (HOST-MEMORY 1, REFUSED 11, ok 460).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.603e-09 |
| algos | adamax | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.17e-09 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002683; torch-eager-bf16 0.000000; torch-compile-bf16 0.002683 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.148e-08; torch-eager-bf16 0.000000; torch-compile-bf16 6.148e-08 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001856; torch-eager-bf16 0.000000; torch-compile-bf16 0.001856 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 7.706e-08; torch-eager-bf16 0.000000; torch-compile-bf16 7.706e-08 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | - | 0.794050 | sklearn-cpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | - | 5.350625 | sklearn-cpu 4.278741 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | 0.755560 | sklearn-cpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | 0.557803 | sklearn-cpu 0.557802 |
| algos | categorical-nb | istella | accuracy (higher is better) | - | 0.838850 | sklearn-cpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | - | 0.412625 | sklearn-cpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | 0.765850 | sklearn-cpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | 0.538866 | sklearn-cpu 0.538866 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | torch-gpu 1.044e-07; numpy-cpu 3.928e-08 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | - | 1.000000 | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000; torch-eager-bf16 1.000000; torch-compile-bf16 1.000000 |
| algos | complement-nb | istella | accuracy (higher is better) | - | 0.849360 | sklearn-cpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | - | 3.762524 | sklearn-cpu 3.174763 |
| algos | complement-nb | taxi | accuracy (higher is better) | - | 0.678020 | sklearn-cpu 0.678030 |
| algos | complement-nb | taxi | logloss (lower is better) | - | 0.715492 | sklearn-cpu 0.715493 |
| algos | complement-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | 0.559491 | sklearn-cpu 0.557285 |
| algos | connected-components | istella | n_components | - | 81 | networkx-cpu 81 |
| algos | connected-components | taxi | n_components | - | 588 | networkx-cpu 588 |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3418.035805; torch-compile-bf16 3418.035805 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.004377; torch-compile-bf16 0.004377 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3417.825326; torch-compile-bf16 3417.825326 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.004438; torch-compile-bf16 0.004438 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | - | 13.931153 | statsmodels-cpu 26.588738; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | - | 96.690449 | statsmodels-cpu 196.955273; statsforecast-cpu 96.685568 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 5.95e-05 | torch-gpu 1.254e-06; numpy-cpu 3.49e-08 |
| algos | eigh | synthetic | relative_residual | - | 5.251e-05 | torch-gpu 1.269e-06; numpy-cpu 2.824e-08 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | enet-cv | istella | r2 (higher is better) | - | 0.326805 | sklearn-cpu 0.326805 |
| algos | enet-cv | istella | rmse (lower is better) | - | 0.685379 | sklearn-cpu 0.685379 |
| algos | enet-cv | taxi | r2 (higher is better) | - | 0.909004 | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | - | 4.804486 | sklearn-cpu 4.804486 |
| algos | gaussian-nb | istella | accuracy (higher is better) | - | 0.876570 | sklearn-cpu 0.876530 |
| algos | gaussian-nb | istella | logloss (lower is better) | - | 3.574420 | sklearn-cpu 3.417392 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | 0.719820 | sklearn-cpu 0.719900 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | 1.132249 | sklearn-cpu 1.133898 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.025557; torch-eager-bf16 2662.412698; torch-compile-bf16 2662.401917 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.36e-08; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001702; torch-eager-bf16 1509.509399; torch-compile-bf16 1509.509515 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 7.846e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.014597; torch-eager-bf16 0.000000; torch-compile-bf16 0.014597 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.508e-08; torch-eager-bf16 0.000000; torch-compile-bf16 9.508e-08 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.140845; torch-eager-bf16 3907.144070; torch-compile-bf16 3678.232431 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 8.05e-08; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.162162; torch-eager-bf16 4680.142857; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 8.776e-08; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | 0.971625 | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-bf16 0.971788; torch-compile-bf16 0.971788 |
| algos | gru-clf | synthetic | logloss (lower is better) | - | 0.070054 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | 0.865723 | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-bf16 0.865777; torch-compile-bf16 0.865777 |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | - | 0.303537 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | - | 0.982337 | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-bf16 0.981940; torch-compile-bf16 0.981940 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | 0.153975 | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-bf16 0.155700; torch-compile-bf16 0.155700 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | 0.747875 | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-bf16 0.748254; torch-compile-bf16 0.748254 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | 0.544554 | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-bf16 0.544144; torch-compile-bf16 0.544144 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | 0.802100 | faiss-cpu 0.801550 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.981250 | faiss-cpu 0.979450 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | 8.307e-08 | shap-cpu 8.63e-15 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | 1.056e-07 | shap-cpu 8.149e-15 |
| algos | knn-imputer | istella | masked_rmse | - | 323953.237332 | sklearn-cpu 986208.700423 |
| algos | knn-imputer | taxi | masked_rmse | - | 6.151696 | sklearn-cpu 5.263919 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.325506 | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.686040 | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909038 | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803593 | sklearn-cpu 4.803593 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.006116; torch-eager-bf16 0.000000; torch-compile-bf16 0.006116 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.286e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.286e-08 |
| algos | lda-clf | istella | accuracy (higher is better) | - | 0.912830 | sklearn-cpu 0.899520 |
| algos | lda-clf | istella | logloss (lower is better) | - | 0.235875 | sklearn-cpu 0.439049 |
| algos | lda-clf | taxi | accuracy (higher is better) | - | 0.762530 | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | - | 0.539763 | sklearn-cpu 0.539767 |
| algos | louvain | istella | modularity | - | 0.911187 | networkx-cpu 0.908460 |
| algos | louvain | istella | n_communities | - | 40 | networkx-cpu 40 |
| algos | louvain | taxi | modularity | - | 0.941953 | networkx-cpu 0.940781 |
| algos | louvain | taxi | n_communities | - | 58 | networkx-cpu 56 |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | 0.967068 | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-clf | synthetic | logloss (lower is better) | - | 0.079720 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | 0.870443 | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-bf16 0.868056; torch-compile-bf16 0.868056 |
| algos | lstm-clf | taxi-hourly | logloss (lower is better) | - | 0.297146 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | 0.979398 | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-bf16 0.981004; torch-compile-bf16 0.981004 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | 0.166295 | torch-eager-fp32 0.159641; torch-compile-fp32 0.159641; torch-eager-bf16 0.159679; torch-compile-bf16 0.159679 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | 0.754306 | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-bf16 0.751591; torch-compile-bf16 0.751591 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | 0.537563 | torch-eager-fp32 0.540429; torch-compile-fp32 0.540429; torch-eager-bf16 0.540526; torch-compile-bf16 0.540526 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | torch-gpu nan; numpy-cpu 0.876106 |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | torch-gpu 0.756366; numpy-cpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 4.003e-07; scipy-cpu 3.439e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 4.041e-07; numpy-cpu 3.259e-08 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.044703; torch-eager-bf16 22862.630675; torch-compile-bf16 22880.029448 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 2.011e-07; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | multinomial-nb | istella | accuracy (higher is better) | - | 0.853620 | sklearn-cpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | - | 3.628565 | sklearn-cpu 3.087499 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | - | 0.723160 | sklearn-cpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | - | 0.590725 | sklearn-cpu 0.590725 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | sklearn-cpu 0.557319 |
| algos | nadam | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.464e-07 |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.438855 | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.150860 | statsforecast-cpu 49.356608 |
| algos | pagerank | istella | sum | - | 1.000000 | networkx-cpu 1.000000 |
| algos | pagerank | taxi | sum | - | 1.000000 | networkx-cpu 1.000000 |
| algos | permutation-shap | istella | rel_error_vs_exact | - | 6.68e-08 | shap-cpu 3.692e-10 |
| algos | permutation-shap | taxi | rel_error_vs_exact | - | 1.054e-07 | shap-cpu 1.218e-15 |
| algos | qr | istella | relative_gram_difference | - | 1.485e-07 | torch-gpu 0.0001698; numpy-cpu 2.462e-08 |
| algos | qr | taxi | relative_gram_difference | - | 1.714e-07 | torch-gpu 6.794e-07; numpy-cpu 3.024e-08 |
| algos | quantile | istella | r2 (higher is better) | - | -0.039998 | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | - | 0.851877 | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | - | 0.899596 | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | - | 5.046749 | sklearn-cpu 5.044706 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | 0.0002359 | torch-gpu 0.0002359; sklearn-cpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | 0.027197 | torch-gpu 0.027197; sklearn-cpu 0.027197 |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.834465; torch-eager-bf16 28679.370880; torch-compile-bf16 18454.670906 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.645e-07; torch-eager-bf16 0.004232; torch-compile-bf16 0.003485 |
| algos | ridge-cv | istella | r2 (higher is better) | - | 0.328684 | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | 0.684422 | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | 0.908981 | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | 4.805109 | sklearn-cpu 4.805057 |
| algos | rmsprop | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.375e-09 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | 0.967828 | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-bf16 0.953559; torch-compile-bf16 0.953559 |
| algos | rnn-clf | synthetic | logloss (lower is better) | - | 0.079029 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | 0.862684 | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-bf16 0.867947; torch-compile-bf16 0.867947 |
| algos | rnn-clf | taxi-hourly | logloss (lower is better) | - | 0.313008 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | r2 (higher is better) | - | 0.978602 | torch-eager-fp32 0.977348; torch-compile-fp32 0.977348; torch-eager-bf16 0.977371; torch-compile-bf16 0.977371 |
| algos | rnn-reg | synthetic | rmse (lower is better) | - | 0.169476 | torch-eager-fp32 0.174374; torch-compile-fp32 0.174374; torch-eager-bf16 0.174283; torch-compile-bf16 0.174283 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | 0.743904 | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-bf16 0.739017; torch-compile-bf16 0.739017 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | 0.548825 | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-bf16 0.554037; torch-compile-bf16 0.554037 |
| algos | sgd-clf | istella | accuracy (higher is better) | - | 0.920330 | sklearn-cpu 0.910200 |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | 0.755330 | sklearn-cpu 0.752520 |
| algos | svd | istella | max_rel_singular_value_error | - | 20.498972 | torch-gpu 2.707e+08; numpy-cpu 1.000000 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | 3.379e-05 | torch-gpu 0.026534; numpy-cpu 4.1e-08 |
| algos | svd | taxi | max_rel_singular_value_error | - | 7.036e-07 | torch-gpu 4.401e-05; numpy-cpu 4.308e-08 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | 1.18e-06 | torch-gpu 0.003043; numpy-cpu 4.314e-08 |
| algos | svgp | istella | r2 (higher is better) | - | -0.106016 | gpytorch-gpu -0.106040; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | 0.878373 | gpytorch-gpu 0.878383; gpytorch-cpu 0.878383 |
| algos | theta | synthetic | forecast_rmse (lower is better) | - | 1.436610 | statsforecast-cpu 1.436557; statsmodels-cpu 1.434862 |
| algos | theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.020604 | statsforecast-cpu 49.253901; statsmodels-cpu 49.311757 |
| algos | tree-shap | istella | max_additivity_error | - | 1.175e-06 | shap-cpu 1.837e-06; xgboost-cpu 1.837e-06; lightgbm-cpu 4.441e-15 |
| algos | tree-shap | taxi | max_additivity_error | - | 3.858e-05 | shap-cpu 0.0001201; xgboost-cpu 0.0001201; lightgbm-cpu 5.684e-13 |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.992124 | sklearn-cpu 0.992170 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.998921 | sklearn-cpu 0.998823 |
| classical | dbscan | istella | n_clusters | - | 40131 | sklearn-cpu 40131 |
| classical | dbscan | istella | noise_fraction | - | 0.219391 | sklearn-cpu 0.219391 |
| classical | dbscan | istella | rows | - | 1000000 | sklearn-cpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | taxi | n_clusters | - | 36 | sklearn-cpu - |
| classical | dbscan | taxi | noise_fraction | - | 0.000174 | sklearn-cpu - |
| classical | dbscan | taxi | rows | - | 1000000 | sklearn-cpu - |
| classical | hdbscan | istella | n_clusters | - | 47 | sklearn-cpu 52 |
| classical | hdbscan | istella | noise_fraction | - | 0.253810 | sklearn-cpu 0.252570 |
| classical | hdbscan | istella | rows | - | 100000 | sklearn-cpu 100000 |
| classical | hdbscan | taxi | n_clusters | - | 161 | sklearn-cpu 161 |
| classical | hdbscan | taxi | noise_fraction | - | 0.140550 | sklearn-cpu 0.134630 |
| classical | hdbscan | taxi | rows | - | 100000 | sklearn-cpu 100000 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | torch-gpu 5.991e+17; sklearn-cpu 6.049e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | torch-gpu -; sklearn-cpu 0.999759 |
| classical | kmeans | istella | n_iter | - | 33 | torch-gpu 91; sklearn-cpu 24 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | torch-gpu 3.06e+08; sklearn-cpu 3.093e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | torch-gpu -; sklearn-cpu 0.999937 |
| classical | kmeans | taxi | n_iter | - | 91 | torch-gpu 54; sklearn-cpu 58 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | torch-gpu 0.978680; sklearn-cpu 1.000000 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | torch-gpu 0; sklearn-cpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | torch-gpu 0.999730; sklearn-cpu 1.000000 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | torch-gpu 0; sklearn-cpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | torch-gpu nan; torch-gpu-eigh 0.151604; sklearn-cpu 0.001881 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | torch-gpu nan; torch-gpu-eigh 0.768589; sklearn-cpu 0.833655 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908836 | torch-gpu 0.908840; torch-gpu-eigh 0.908822; sklearn-cpu 0.724850 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696479 | torch-gpu 4.696376; torch-gpu-eigh 4.696849; sklearn-cpu 8.159187 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | torch-gpu 1.000000; sklearn-cpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999997 | torch-gpu 0.999997; sklearn-cpu 0.999995 |
| classical2 | elasticnet | istella | r2 (higher is better) | - | 0.260922 | sklearn-cpu 0.260922 |
| classical2 | elasticnet | istella | rmse (lower is better) | - | 0.718134 | sklearn-cpu 0.718134 |
| classical2 | elasticnet | taxi | r2 (higher is better) | - | 0.907378 | sklearn-cpu 0.907378 |
| classical2 | elasticnet | taxi | rmse (lower is better) | - | 4.847225 | sklearn-cpu 4.847224 |
| classical2 | gmm | istella | bic (lower is better) | - | -3.851e+07 | - |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | - | 200.794500 | - |
| classical2 | gmm | istella | n_iter | - | 24 | - |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.668e+06 | - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.807640 | - |
| classical2 | gmm | taxi | n_iter | - | 29 | - |
| classical2 | ivf | istella | recall_at_k (higher is better) | - | 0.999925 | faiss-cpu - |
| classical2 | ivf | istella | rows_with_repeated_ids | - | 0 | faiss-cpu - |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | 0.999650 | faiss-cpu - |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | 0 | faiss-cpu - |
| classical2 | lasso | istella | r2 (higher is better) | - | 0.310837 | sklearn-cpu 0.310837 |
| classical2 | lasso | istella | rmse (lower is better) | - | 0.693460 | sklearn-cpu 0.693460 |
| classical2 | lasso | taxi | r2 (higher is better) | - | 0.908995 | sklearn-cpu 0.908995 |
| classical2 | lasso | taxi | rmse (lower is better) | - | 4.804744 | sklearn-cpu 4.804745 |
| classical2 | logreg | istella | accuracy (higher is better) | - | 0.924590 | sklearn-cpu 0.924590 |
| classical2 | logreg | istella | logloss (lower is better) | - | 0.181249 | sklearn-cpu 0.181264 |
| classical2 | logreg | istella | nonfinite_proba_rows | - | 0 | sklearn-cpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | - | 0.763350 | sklearn-cpu 0.763320 |
| classical2 | logreg | taxi | logloss (lower is better) | - | 0.538985 | sklearn-cpu 0.538980 |
| classical2 | logreg | taxi | nonfinite_proba_rows | - | 0 | sklearn-cpu 0 |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328674 | sklearn-cpu 0.328676 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684427 | sklearn-cpu 0.684426 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | sklearn-cpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805050 | sklearn-cpu 4.805056 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | sklearn-cpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | - | 0.0001314 | sklearn-cpu 0.000122 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999965 | sklearn-cpu 0.999965 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | - | 0.003257 | sklearn-cpu 0.003257 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | torch-eager-fp32 2.802e-06; torch-compile-fp32 2.802e-06; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-bf16 9.018647; torch-compile-fp32 9.018733; torch-compile-bf16 9.018664 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | torch-eager-fp32 9.018732; torch-compile-fp32 9.018733; torch-eager-bf16 9.018646; torch-compile-bf16 9.018663 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.418446 | torch-eager-fp32 8.418447; torch-compile-fp32 8.418446; torch-eager-bf16 8.416107; torch-compile-bf16 8.415815 |
| neural | lm-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | torch-eager-fp32 1.160401; torch-compile-fp32 1.160401; torch-eager-bf16 1.160498; torch-compile-bf16 1.160367 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | torch-eager-fp32 1.123361; torch-compile-fp32 1.123361; torch-eager-bf16 1.123461; torch-compile-bf16 1.123438 |
| neural | mlp-train-step | gaussian | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | torch-eager-fp32 5.635910; torch-compile-fp32 5.635910; torch-eager-bf16 5.635892; torch-compile-bf16 5.635953 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | torch-eager-fp32 5.635909; torch-compile-fp32 -; torch-eager-bf16 5.635892; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | torch-eager-fp32 4.833934; torch-compile-fp32 -; torch-eager-bf16 4.834043; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 -; torch-eager-bf16 2; torch-compile-bf16 - |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | 0.631297 | xgboost-gpu -; catboost-cpu 0.628772; xgboost-cpu 0.631473; lightgbm-cpu 0.632665 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | 0.528285 | xgboost-gpu -; catboost-cpu 0.528923; xgboost-cpu 0.528686; lightgbm-cpu 0.528094 |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.983304 | xgboost-gpu -; catboost-cpu 0.983135; xgboost-cpu 0.983622 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.156072 | xgboost-gpu -; catboost-cpu 0.157692; xgboost-cpu 0.149263 |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.632205 | xgboost-gpu -; catboost-cpu 0.632578; xgboost-cpu 0.630968 |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.527838 | xgboost-gpu -; catboost-cpu 0.527851; xgboost-cpu 0.528677 |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983586 | xgboost-gpu -; catboost-cpu 0.983135; xgboost-cpu 0.983622; lightgbm-cpu 0.983778 |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.150066 | xgboost-gpu -; catboost-cpu 0.157692; xgboost-cpu 0.149263; lightgbm-cpu 0.149653 |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.632096 | xgboost-gpu -; catboost-cpu 0.632578; xgboost-cpu 0.630968; lightgbm-cpu 0.632243 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528044 | xgboost-gpu -; catboost-cpu 0.527851; xgboost-cpu 0.528677; lightgbm-cpu 0.528067 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.907556 | xgboost-gpu -; catboost-cpu 0.907768; xgboost-cpu 0.910140; lightgbm-cpu 0.910058 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.258413 | xgboost-gpu -; catboost-cpu 0.258286; xgboost-cpu 0.246803; lightgbm-cpu 0.245916 |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.599380 | xgboost-gpu -; catboost-cpu 0.599150; xgboost-cpu 0.601128; lightgbm-cpu 0.601580 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.012595 | xgboost-gpu -; catboost-cpu 1.012734; xgboost-cpu 1.005128; lightgbm-cpu 1.004282 |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.979444 | catboost-cpu 0.979221 |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.190928 | catboost-cpu 0.192114 |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.629203 | catboost-cpu 0.628918 |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.529007 | catboost-cpu 0.529083 |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.854545 | xgboost-gpu -; catboost-cpu 0.846328; xgboost-cpu 0.872796 |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.719953 | xgboost-gpu -; catboost-cpu 0.713361; xgboost-cpu 0.738397 |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.650400 | xgboost-gpu -; catboost-cpu 0.643611; xgboost-cpu 0.670093 |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-cpu 0.979899 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-cpu 0.188093 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-cpu 0.630269 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-cpu 0.528650 |


## Neural headline: ours IDENTICAL against torch bf16

What customers run is torch in bf16. The headline divides our IDENTICAL median by torch's fastest bf16 arm (eager or compile, the lower stored median); the fp32 twin (torch's fastest fp32 arm) is the second column. Per-arm ratios stay in each race's table below.

| lane | dataset | ours IDENTICAL ms | torch bf16 ms (fastest arm) | ours / torch bf16 | torch fp32 twin ms (fastest arm) | ours / torch fp32 | note |
|---|---|---|---|---|---|---|---|
| adafactor | synthetic | 8.5 | - | - | 7.5 (torch-eager-fp32) | 1.131 | no torch bf16 arm on this lane |
| adagrad | synthetic | 3.3 | - | - | 4.7 (torch-eager-fp32) | 0.696 | no torch bf16 arm on this lane |
| adamax | synthetic | 3.7 | - | - | 5.7 (torch-eager-fp32) | 0.649 | no torch bf16 arm on this lane |
| avgpool1d | synthetic | 0.5 | 0.6 (torch-compile-bf16) | 0.773 | 0.7 (torch-eager-fp32) | 0.669 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| avgpool2d | synthetic | 0.7 | 0.9 (torch-compile-bf16) | 0.794 | 1.0 (torch-eager-fp32) | 0.746 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm1d | synthetic | 2.1 | 0.9 (torch-eager-bf16) | 2.242 | 0.8 (torch-eager-fp32) | 2.478 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm2d | synthetic | 2.2 | 0.7 (torch-eager-bf16) | 3.006 | 1.0 (torch-eager-fp32) | 2.237 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| cnn-clf | synthetic | 314.2 | 251.7 (torch-eager-bf16) | 1.248 | 172.9 (torch-eager-fp32) | 1.817 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv1d | synthetic | 7.3 | 1.3 (torch-eager-bf16) | 5.777 | 2.0 (torch-eager-fp32) | 3.602 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv2d | synthetic | 6.0 | 1.1 (torch-eager-bf16) | 5.747 | 1.1 (torch-eager-fp32) | 5.629 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| dropout2d | synthetic | 0.4 | - | - | 0.7 (torch-eager-fp32) | 0.506 | no torch bf16 arm on this lane |
| embedding | synthetic | 2.0 | - | - | 1.0 (torch-compile-fp32) | 2.094 | no torch bf16 arm on this lane |
| gcn | istella | 3.0 | 3.3 (torch-compile-bf16) | 0.921 | 3.6 (torch-compile-fp32) | 0.834 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gcn | taxi | 1.4 | 2.8 (torch-compile-bf16) | 0.491 | 3.2 (torch-compile-fp32) | 0.428 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-avgpool | synthetic | 1.2 | 0.4 (torch-eager-bf16) | 2.661 | 0.5 (torch-eager-fp32) | 2.309 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-maxpool | synthetic | 1.2 | 0.6 (torch-eager-bf16) | 1.934 | 0.5 (torch-eager-fp32) | 2.535 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | istella | 5.9 | 3.2 (torch-compile-bf16) | 1.843 | 3.4 (torch-compile-fp32) | 1.756 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | taxi | 1.6 | 1.4 (torch-compile-bf16) | 1.190 | 1.4 (torch-eager-fp32) | 1.157 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | synthetic | 514.9 | 1852.7 (torch-eager-bf16) | 0.278 | 1946.7 (torch-eager-fp32) | 0.265 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | taxi-hourly | 515.5 | 1899.7 (torch-compile-bf16) | 0.271 | 1792.8 (torch-eager-fp32) | 0.288 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | synthetic | 513.3 | 2078.2 (torch-eager-bf16) | 0.247 | 1993.7 (torch-eager-fp32) | 0.257 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | taxi-hourly | 513.4 | 1801.3 (torch-eager-bf16) | 0.285 | 2006.3 (torch-compile-fp32) | 0.256 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lamb | synthetic | 13.4 | - | - | - | - | no torch bf16 arm on this lane |
| layernorm | synthetic | 1.2 | 0.9 (torch-eager-bf16) | 1.319 | 0.8 (torch-eager-fp32) | 1.546 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lion | synthetic | 3.4 | - | - | - | - | no torch bf16 arm on this lane |
| lstm-clf | synthetic | 631.6 | 854.9 (torch-eager-bf16) | 0.739 | 894.9 (torch-compile-fp32) | 0.706 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-clf | taxi-hourly | 630.7 | 977.6 (torch-compile-bf16) | 0.645 | 815.7 (torch-compile-fp32) | 0.773 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | synthetic | 628.6 | 1053.4 (torch-eager-bf16) | 0.597 | 923.1 (torch-eager-fp32) | 0.681 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | taxi-hourly | 627.9 | 1017.5 (torch-eager-bf16) | 0.617 | 920.3 (torch-eager-fp32) | 0.682 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool1d | synthetic | 0.4 | 0.8 (torch-eager-bf16) | 0.572 | 0.8 (torch-compile-fp32) | 0.525 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool2d | synthetic | 0.6 | 1.0 (torch-eager-bf16) | 0.659 | 0.9 (torch-eager-fp32) | 0.686 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| moe | synthetic | 15.2 | 3.2 (torch-compile-bf16) | 4.731 | 5.6 (torch-eager-fp32) | 2.709 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| nadam | synthetic | 4.0 | - | - | 7.0 (torch-eager-fp32) | 0.575 | no torch bf16 arm on this lane |
| resnet-block | synthetic | 15.0 | 2.1 (torch-eager-bf16) | 7.301 | 2.2 (torch-eager-fp32) | 6.811 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rmsprop | synthetic | 3.4 | - | - | 5.1 (torch-eager-fp32) | 0.659 | no torch bf16 arm on this lane |
| rnn-clf | synthetic | 305.9 | 950.8 (torch-eager-bf16) | 0.322 | 879.7 (torch-compile-fp32) | 0.348 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-clf | taxi-hourly | 306.6 | 975.1 (torch-eager-bf16) | 0.314 | 743.5 (torch-eager-fp32) | 0.412 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | synthetic | 300.1 | 898.5 (torch-eager-bf16) | 0.334 | 891.9 (torch-eager-fp32) | 0.337 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | taxi-hourly | 300.7 | 1010.8 (torch-eager-bf16) | 0.297 | 762.7 (torch-compile-fp32) | 0.394 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gemm | gaussian | 28.5 | 11.3 (torch-eager-bf16) | 2.528 | 13.5 (torch-eager-fp32) | 2.117 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-forward | bytes | 34.7 | 13.3 (torch-eager-bf16) | 2.613 | 14.5 (torch-compile-fp32) | 2.392 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-train-step | bytes | 42.8 | 8.2 (torch-compile-bf16) | 5.236 | 15.2 (torch-compile-fp32) | 2.820 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mlp-train-step | gaussian | 3.4 | 1.5 (torch-compile-bf16) | 2.309 | 1.3 (torch-eager-fp32) | 2.583 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-forward | bytes | 5.6 | 5.7 (torch-compile-bf16) | 0.994 | 5.9 (torch-compile-fp32) | 0.953 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-train-step | bytes | 116.7 | 127.6 (torch-eager-bf16) | 0.915 | 135.2 (torch-eager-fp32) | 0.863 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| transformer-forward | gaussian | 3.2 | 1.9 (torch-eager-bf16) | 1.661 | 1.6 (torch-compile-fp32) | 2.047 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |

## Trees

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 10258.4 | 10258.4..10258.4 | 1 | - | - | 10258.4 | - | - (stored whole) | - | - | - | - | auc=0.631297, logloss=0.528285 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 97857.4 | 97857.4..97857.4 | 1 | 0.105 | - | 97857.4 | 97857.4 | 0.00 (cpu-arm) | 0.105 (whole/whole) | - | 11918.4 | - | auc=0.628772, logloss=0.528923 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 18924.8 | 18924.8..18924.8 | 1 | 0.542 | - | 18924.8 | 18924.8 | 0.00 (cpu-arm) | 0.542 (whole/whole) | - | 8838.5 | - | auc=0.631473, logloss=0.528686 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 10928.0 | 10928.0..10928.0 | 1 | 0.939 | - | 10928.0 | 10928.0 | 0.00 (cpu-arm) | 0.939 (whole/whole) | - | 8741.9 | - | auc=0.632665, logloss=0.528094 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 6295.4 | 6295.4..6295.4 | 1 | - | - | 6295.4 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 70785.7 | 70785.7..70785.7 | 1 | 0.089 | - | 70785.7 | 70785.7 | 0.00 (cpu-arm) | 0.089 (whole/whole) | - | 10524.7 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 16034.8 | 16034.8..16034.8 | 1 | 0.393 | - | 16034.8 | 16034.8 | 0.00 (cpu-arm) | 0.393 (whole/whole) | - | 11639.5 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3631.3 | 3631.3..3631.3 | 1 | - | - | 3631.3 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 43791.1 | 43791.1..43791.1 | 1 | 0.083 | - | 43791.1 | 43791.1 | 0.00 (cpu-arm) | 0.083 (whole/whole) | - | 7279.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 8079.8 | 8079.8..8079.8 | 1 | 0.449 | - | 8079.8 | 8079.8 | 0.00 (cpu-arm) | 0.449 (whole/whole) | - | 7413.2 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 6462.9 | 6462.9..6462.9 | 1 | - | - | 6462.9 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 103179.7 | 103179.7..103179.7 | 1 | 0.063 | - | 103179.7 | 103179.7 | 0.00 (cpu-arm) | 0.063 (whole/whole) | - | 10528.0 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 23613.1 | 23613.1..23613.1 | 1 | 0.274 | - | 23613.1 | 23613.1 | 0.00 (cpu-arm) | 0.274 (whole/whole) | - | 11234.1 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 19843.3 | 19843.3..19843.3 | 1 | 0.326 | - | 19843.3 | 19843.3 | 0.00 (cpu-arm) | 0.326 (whole/whole) | - | 11308.2 | - | auc=0.983778, logloss=0.149653 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3662.6 | 3662.6..3662.6 | 1 | - | - | 3662.6 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 77432.0 | 77432.0..77432.0 | 1 | 0.047 | - | 77432.0 | 77432.0 | 0.00 (cpu-arm) | 0.047 (whole/whole) | - | 6674.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 10350.4 | 10350.4..10350.4 | 1 | 0.354 | - | 10350.4 | 10350.4 | 0.00 (cpu-arm) | 0.354 (whole/whole) | - | 6853.6 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 10099.2 | 10099.2..10099.2 | 1 | 0.363 | - | 10099.2 | 10099.2 | 0.00 (cpu-arm) | 0.363 (whole/whole) | - | 6933.8 | - | auc=0.632243, logloss=0.528067 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 10051.4 | 10051.4..10051.4 | 1 | - | - | 10051.4 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 561543.4 | 561543.4..561543.4 | 1 | 0.018 | - | 561543.4 | 561543.4 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 11547.8 | - | accuracy=0.907768, mlogloss=0.258286 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 76053.9 | 76053.9..76053.9 | 1 | 0.132 | - | 76053.9 | 76053.9 | 0.00 (cpu-arm) | 0.132 (whole/whole) | - | 12606.4 | - | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 79078.6 | 79078.6..79078.6 | 1 | 0.127 | - | 79078.6 | 79078.6 | 0.00 (cpu-arm) | 0.127 (whole/whole) | - | 12549.6 | - | accuracy=0.910058, mlogloss=0.245916 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 6049.8 | 6049.8..6049.8 | 1 | - | - | 6049.8 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 178349.9 | 178349.9..178349.9 | 1 | 0.034 | - | 178349.9 | 178349.9 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 8021.3 | - | accuracy=0.599150, mlogloss=1.012734 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 35861.4 | 35861.4..35861.4 | 1 | 0.169 | - | 35861.4 | 35861.4 | 0.00 (cpu-arm) | 0.169 (whole/whole) | - | 8463.4 | - | accuracy=0.601128, mlogloss=1.005128 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 41539.9 | 41539.9..41539.9 | 1 | 0.146 | - | 41539.9 | 41539.9 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 8711.3 | - | accuracy=0.601580, mlogloss=1.004282 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 34452.1 | 34452.1..34452.1 | 1 | - | - | 34452.1 | - | - (stored whole) | - | - | - | - | auc=0.979444, logloss=0.190928 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 273122.4 | 273122.4..273122.4 | 1 | 0.126 | - | 273122.4 | 273122.4 | 0.00 (cpu-arm) | 0.126 (whole/whole) | - | 11159.1 | - | auc=0.979221, logloss=0.192114 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 19884.4 | 19884.4..19884.4 | 1 | - | - | 19884.4 | - | - (stored whole) | - | - | - | - | auc=0.629203, logloss=0.529007 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 78566.1 | 78566.1..78566.1 | 1 | 0.253 | - | 78566.1 | 78566.1 | 0.00 (cpu-arm) | 0.253 (whole/whole) | - | 8753.0 | - | auc=0.628918, logloss=0.529083 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 1541.0 | 1541.0..1541.0 | 1 | - | - | 1541.0 | - | - (stored whole) | - | - | - | - | map=0.854545, ndcg10=0.719953, ndcg5=0.650400 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 31176.3 | 31176.3..31176.3 | 1 | 0.049 | - | 31176.3 | 31176.3 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 12433.4 | - | map=0.846328, ndcg10=0.713361, ndcg5=0.643611 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 6075.9 | 6075.9..6075.9 | 1 | 0.254 | - | 6075.9 | 6075.9 | 0.00 (cpu-arm) | 0.254 (whole/whole) | - | 13581.1 | - | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 5995.7 | 5995.7..5995.7 | 1 | - | - | 5995.7 | - | - (stored whole) | - | - | - | - | auc=0.980129, logloss=0.186686 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 44398.5 | 44398.5..44398.5 | 1 | 0.135 | - | 44398.5 | 44398.5 | 0.00 (cpu-arm) | 0.135 (whole/whole) | - | 9400.6 | - | auc=0.979899, logloss=0.188093 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3958.8 | 3958.8..3958.8 | 1 | - | - | 3958.8 | - | - (stored whole) | - | - | - | - | auc=0.630436, logloss=0.528554 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 21286.9 | 21286.9..21286.9 | 1 | 0.186 | - | 21286.9 | 21286.9 | 0.00 (cpu-arm) | 0.186 (whole/whole) | - | 5984.4 | - | auc=0.630269, logloss=0.528650 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1558)`, ran on AMD Instinct MI325X (amd, DO) job a1558

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 54245.1 | 54245.1..54245.1 | 1 | - | - | 54245.1 | - | - (stored whole) | - | - | - | - | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1558 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 263657.7 | 263657.7..263657.7 | 1 | 0.206 | - | 263657.7 | 263657.7 | 0.00 (cpu-arm) | 0.206 (whole/whole) | - | 5971.6 | - | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1558)`, ran on AMD Instinct MI325X (amd, DO) job a1558

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 7056.5 | 7056.5..7056.5 | 1 | - | - | 7056.5 | - | - (stored whole) | - | - | - | - | n_clusters=36, noise_fraction=0.000174, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1558 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | HOST-MEMORY(killed at 143.0 GB: the driver's process tree held 153.6 GB, over 90% of the box's 168.8 GB) (copied from release-board-resume-r2; measured this run) |

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1163)`, ran on AMD Instinct MI325X (amd, DO) job a1163

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 1348.5 | 1348.5..1348.5 | 1 | - | - | 1348.5 | - | - (stored whole) | - | - | - | - | n_clusters=47, noise_fraction=0.253810, rows=100000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 829354.4 | 829354.4..829354.4 | 1 | 0.002 | - | 829354.4 | 829354.4 | 0.00 (cpu-arm) | 0.002 (whole/whole) | - | 1462.1 | - | n_clusters=52, noise_fraction=0.252570, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1163)`, ran on AMD Instinct MI325X (amd, DO) job a1163

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 387.8 | 387.8..387.8 | 1 | - | - | 387.8 | - | - (stored whole) | - | - | - | - | n_clusters=161, noise_fraction=0.140550, rows=100000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38362.1 | 38362.1..38362.1 | 1 | 0.010 | - | 38362.1 | 38362.1 | 0.00 (cpu-arm) | 0.010 (whole/whole) | - | 334.0 | - | n_clusters=161, noise_fraction=0.134630, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1163)`, ran on AMD Instinct MI325X (amd, DO) job a1163

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 296.1 | 296.1..296.1 | 1 | - | - | 296.1 | - | - (stored whole) | - | - | - | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3406.3 | 3406.3..3406.3 | 1 | 0.087 | - | 3592.3 | 3406.3 | 186.04 (upload_ms_untimed) | 0.082 (whole/whole (kernel not derivable)) | - | 5132.4 | 3460.8 | inertia=5.991e+17, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 5614.6 | 5614.6..5614.6 | 1 | 0.053 | - | 5614.6 | 5614.6 | 0.00 (cpu-arm) | 0.053 (whole/whole) | - | 5794.7 | - | inertia=6.049e+17, inertia_over_ours=0.999759, n_iter=24 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:34:26Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1163)`, ran on AMD Instinct MI325X (amd, DO) job a1163

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 187.2 | 187.2..187.2 | 1 | - | - | 187.2 | - | - (stored whole) | - | - | - | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 538.0 | 538.0..538.0 | 1 | 0.348 | - | 684.4 | 538.0 | 146.41 (upload_ms_untimed) | 0.274 (whole/whole (kernel not derivable)) | - | 3194.4 | 459.9 | inertia=3.06e+08, n_iter=54 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 2297.6 | 2297.6..2297.6 | 1 | 0.081 | - | 2297.6 | 2297.6 | 0.00 (cpu-arm) | 0.081 (whole/whole) | - | 772.1 | - | inertia=3.093e+08, inertia_over_ours=0.999937, n_iter=58 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:33:52Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn / istella (rows full, shape 400000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1199)`, ran on AMD Instinct MI325X (amd, DO) job a1199

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 30.1 | 30.1..30.1 | 1 | - | - | 30.1 | - | - (stored whole) | - | - | - | - | recall_at_k=0.976250, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1199 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 32.3 | 32.3..32.3 | 1 | 0.931 | - | 181.0 | 32.3 | 148.73 (upload_ms_untimed) | 0.166 (whole/whole (kernel not derivable)) | - | 3187.9 | 3884.6 | recall_at_k=0.978680, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 536.9 | 536.9..536.9 | 1 | 0.056 | - | 536.9 | 536.9 | 0.00 (cpu-arm) | 0.056 (whole/whole) | - | 587.8 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:36:26Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1199)`, ran on AMD Instinct MI325X (amd, DO) job a1199

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 13.2 | 13.2..13.2 | 1 | - | - | 13.2 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999754, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1199 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 27.2 | 27.2..27.2 | 1 | 0.486 | - | 175.2 | 27.2 | 147.94 (upload_ms_untimed) | 0.076 (whole/whole (kernel not derivable)) | - | 2866.0 | 3242.7 | recall_at_k=0.999730, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 381.1 | 381.1..381.1 | 1 | 0.035 | - | 381.1 | 381.1 | 0.00 (cpu-arm) | 0.035 (whole/whole) | - | 238.1 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:36:05Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1193)`, ran on AMD Instinct MI325X (amd, DO) job a1193

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 541.8 | 541.8..541.8 | 1 | - | - | 541.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1193 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2340.8 | 2340.8..2340.8 | 1 | 0.231 | - | 2523.3 | 2340.8 | 182.42 (upload_ms_untimed) | 0.215 (whole/whole (kernel not derivable)) | - | 5648.2 | 5303.6 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 42.8 | 42.8..42.8 | 1 | 12.658 | - | 222.4 | 42.8 | 179.55 (upload_ms_untimed) | 2.437 (whole/whole (kernel not derivable)) | - | 5051.5 | 3649.4 | finite=True, r2=0.151604, rmse=0.768589 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3980.6 | 3980.6..3980.6 | 1 | 0.136 | - | 3980.6 | 3980.6 | 0.00 (cpu-arm) | 0.136 (whole/whole) | - | 5793.0 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:47Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1193)`, ran on AMD Instinct MI325X (amd, DO) job a1193

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 12.5 | 12.5..12.5 | 1 | - | - | 12.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908836, rmse=4.696479 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1193 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 100.7 | 100.7..100.7 | 1 | 0.124 | - | 243.7 | 100.7 | 143.01 (upload_ms_untimed) | 0.051 (whole/whole (kernel not derivable)) | - | 3535.9 | 695.3 | finite=True, r2=0.908840, rmse=4.696376 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 34.0 | 34.0..34.0 | 1 | 0.367 | - | 176.6 | 34.0 | 142.61 (upload_ms_untimed) | 0.071 (whole/whole (kernel not derivable)) | - | 3018.3 | 572.0 | finite=True, r2=0.908822, rmse=4.696849 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 304.7 | 304.7..304.7 | 1 | 0.041 | - | 304.7 | 304.7 | 0.00 (cpu-arm) | 0.041 (whole/whole) | - | 782.5 | - | finite=True, r2=0.724850, rmse=8.159187 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:14Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1553)`, ran on AMD Instinct MI325X (amd, DO) job a1553

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 51.8 | 51.8..51.8 | 1 | - | - | 51.8 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000 | - | main board, one scored run | - | ok (main@a8e0548a5 amd/a1553 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 19.9 | 19.9..19.9 | 1 | 2.604 | - | 206.7 | 19.9 | 186.78 (upload_ms_untimed) | 0.251 (whole/whole (kernel not derivable)) | - | 5044.4 | 3505.8 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 621.7 | 621.7..621.7 | 1 | 0.083 | - | 621.7 | 621.7 | 0.00 (cpu-arm) | 0.083 (whole/whole) | - | 2342.9 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:34:57Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1553)`, ran on AMD Instinct MI325X (amd, DO) job a1553

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 7.5 | 7.5..7.5 | 1 | - | - | 7.5 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999997 | - | main board, one scored run | - | ok (main@a8e0548a5 amd/a1553 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 10.9 | 10.9..10.9 | 1 | 0.691 | - | 152.7 | 10.9 | 141.81 (upload_ms_untimed) | 0.049 (whole/whole (kernel not derivable)) | - | 3009.5 | 412.0 | explained_variance_ratio_sum=0.999997 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 82.2 | 82.2..82.2 | 1 | 0.091 | - | 82.2 | 82.2 | 0.00 (cpu-arm) | 0.091 (whole/whole) | - | 401.6 | - | explained_variance_ratio_sum=0.999995 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:34:39Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical, wave 2

### elasticnet / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 66.5 | 66.5..66.5 | 1 | - | - | 66.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.260922, rmse=0.718134 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 2177.9 | 2177.9..2177.9 | 1 | 0.031 | - | 2177.9 | 2177.9 | 0.00 (cpu-arm) | 0.031 (whole/whole) | - | 2803.2 | - | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### elasticnet / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 5.0 | 5.0..5.0 | 1 | - | - | 5.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.907378, rmse=4.847225 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 113.1 | 113.1..113.1 | 1 | 0.044 | - | 113.1 | 113.1 | 0.00 (cpu-arm) | 0.044 (whole/whole) | - | 332.3 | - | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on AMD Instinct MI325X (amd, DO) job a0536

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 541.5 | 541.5..541.5 | 1 | - | - | 541.5 | - | - (stored whole) | - | - | - | - | bic=-3.851e+07, mean_log_likelihood=200.794500, n_iter=24 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on AMD Instinct MI325X (amd, DO) job a0536

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 56.3 | 56.3..56.3 | 1 | - | - | 56.3 | - | - (stored whole) | - | - | - | - | bic=-3.668e+06, mean_log_likelihood=12.807640, n_iter=29 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1163)`, ran on AMD Instinct MI325X (amd, DO) job a1163

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 194.9 | 194.9..194.9 | 1 | - | - | 194.9 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999925, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 3704.2 | 3704.2..3704.2 | 1 | 0.053 | - | 3704.2 | 3704.2 | 0.00 (cpu-arm) | 0.053 (whole/whole) | - | 1286.6 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1163)`, ran on AMD Instinct MI325X (amd, DO) job a1163

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 44.7 | 44.7..44.7 | 1 | - | - | 44.7 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999650, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 143.0 | 143.0..143.0 | 1 | 0.313 | - | 143.0 | 143.0 | 0.00 (cpu-arm) | 0.313 (whole/whole) | - | 139.0 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1199)`, ran on AMD Instinct MI325X (amd, DO) job a1199

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 83.3 | 83.3..83.3 | 1 | - | - | 83.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310837, rmse=0.693460 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1199 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 7075.0 | 7075.0..7075.0 | 1 | 0.012 | - | 7075.0 | 7075.0 | 0.00 (cpu-arm) | 0.012 (whole/whole) | - | 2803.7 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1199)`, ran on AMD Instinct MI325X (amd, DO) job a1199

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 5.0 | 5.0..5.0 | 1 | - | - | 5.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908995, rmse=4.804744 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1199 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 123.8 | 123.8..123.8 | 1 | 0.041 | - | 123.8 | 123.8 | 0.00 (cpu-arm) | 0.041 (whole/whole) | - | 332.5 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### logreg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 2121.1 | 2121.1..2121.1 | 1 | - | - | 2121.1 | - | - (stored whole) | - | - | - | - | accuracy=0.924590, logloss=0.181249, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 20612.3 | 20612.3..20612.3 | 1 | 0.103 | - | 20612.3 | 20612.3 | 0.00 (cpu-arm) | 0.103 (whole/whole) | - | 2833.9 | - | accuracy=0.924590, logloss=0.181264, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### logreg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1161)`, ran on AMD Instinct MI325X (amd, DO) job a1161

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 23.4 | 23.4..23.4 | 1 | - | - | 23.4 | - | - (stored whole) | - | - | - | - | accuracy=0.763350, logloss=0.538985, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1161 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 233.9 | 233.9..233.9 | 1 | 0.100 | - | 233.9 | 233.9 | 0.00 (cpu-arm) | 0.100 (whole/whole) | - | 367.1 | - | accuracy=0.763320, logloss=0.538980, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1193)`, ran on AMD Instinct MI325X (amd, DO) job a1193

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 454.9 | 454.9..454.9 | 1 | - | - | 454.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328674, rmse=0.684427 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1193 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 4288.8 | 4288.8..4288.8 | 1 | 0.106 | - | 4288.8 | 4288.8 | 0.00 (cpu-arm) | 0.106 (whole/whole) | - | 8688.9 | - | finite=True, r2=0.328676, rmse=0.684426 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1193)`, ran on AMD Instinct MI325X (amd, DO) job a1193

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 4.7 | 4.7..4.7 | 1 | - | - | 4.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1193 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 33.2 | 33.2..33.2 | 1 | 0.142 | - | 33.2 | 33.2 | 0.00 (cpu-arm) | 0.142 (whole/whole) | - | 291.3 | - | finite=True, r2=0.908983, rmse=4.805056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1553)`, ran on AMD Instinct MI325X (amd, DO) job a1553

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 30.2 | 30.2..30.2 | 1 | - | - | 30.2 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001314 | - | main board, one scored run | - | ok (main@a8e0548a5 amd/a1553 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 689.8 | 689.8..689.8 | 1 | 0.044 | - | 689.8 | 689.8 | 0.00 (cpu-arm) | 0.044 (whole/whole) | - | 2422.2 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.000122 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1553)`, ran on AMD Instinct MI325X (amd, DO) job a1553

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 3.9 | 3.9..3.9 | 1 | - | - | 3.9 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | main board, one scored run | - | ok (main@a8e0548a5 amd/a1553 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 205.6 | 205.6..205.6 | 1 | 0.019 | - | 205.6 | 205.6 | 0.00 (cpu-arm) | 0.019 (whole/whole) | - | 460.6 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Neural

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1189)`, ran on AMD Instinct MI325X (amd, DO) job a1189

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 11.3 ms) = 2.528; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 13.5 ms) = 2.117. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 28.5 | 28.5..28.5 | 1 | - | - | 28.5 | - | - (stored whole) | - | - | - | - | max_rel_err_vs_fp64=2.399e-07 | - | main board, one scored run | - | ok (main@9f83ea479 amd/a1189 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.5 | 13.5..13.5 | 1 | 2.117 | - | 13.5 | - | - (stored whole) | 2.117 (whole/whole (kernel not derivable)) | - | 2912.1 | 268.0 | max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.9 | 13.9..13.9 | 1 | 2.057 | - | 13.9 | - | - (stored whole) | 2.057 (whole/whole (kernel not derivable)) | - | 3033.6 | 268.0 | max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 11.3 | 11.3..11.3 | 1 | 2.528 | - | 11.3 | - | - (stored whole) | 2.528 (whole/whole (kernel not derivable)) | - | 4298.6 | 300.0 | max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.3 | 13.3..13.3 | 1 | 2.138 | - | 13.3 | - | - (stored whole) | 2.138 (whole/whole (kernel not derivable)) | - | 4532.3 | 300.0 | max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1557)`, ran on AMD Instinct MI325X (amd, DO) job a1557

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 13.3 ms) = 2.613; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 14.5 ms) = 2.392. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 34.7 | 34.7..34.7 | 1 | - | - | 34.7 | - | - (stored whole) | - | - | - | - | mean_nll=9.018733 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1557 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 17.6 | 17.6..17.6 | 1 | 1.972 | - | 17.6 | - | - (stored whole) | 1.972 (whole/whole (kernel not derivable)) | - | 3122.2 | 241.0 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.3 | 13.3..13.3 | 1 | 2.613 | - | 13.3 | - | - (stored whole) | 2.613 (whole/whole (kernel not derivable)) | - | 4342.2 | 240.5 | mean_nll=9.018647 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 14.5 | 14.5..14.5 | 1 | 2.392 | - | 14.5 | - | - (stored whole) | 2.392 (whole/whole (kernel not derivable)) | - | 3217.8 | 222.5 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.8 | 13.8..13.8 | 1 | 2.511 | - | 13.8 | - | - (stored whole) | 2.511 (whole/whole (kernel not derivable)) | - | 4448.6 | 194.5 | mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-bf16, torch-compile-fp32, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1557)`, ran on AMD Instinct MI325X (amd, DO) job a1557

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 8.2 ms) = 5.236; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 15.2 ms) = 2.820. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 42.8 | 42.8..42.8 | 1 | - | - | 42.8 | - | - (stored whole) | - | - | - | - | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1557 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 18.2 | 18.2..18.2 | 1 | 2.349 | - | 18.2 | - | - (stored whole) | 2.349 (whole/whole (kernel not derivable)) | - | 3311.5 | 931.1 | loss_first_step=9.018732, loss_last_step=8.418447, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 15.2 | 15.2..15.2 | 1 | 2.820 | - | 15.2 | - | - (stored whole) | 2.820 (whole/whole (kernel not derivable)) | - | 3359.7 | 780.1 | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 13.0 | 13.0..13.0 | 1 | 3.299 | - | 13.0 | - | - (stored whole) | 3.299 (whole/whole (kernel not derivable)) | - | 4897.8 | 774.1 | loss_first_step=9.018646, loss_last_step=8.416107, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 8.2 | 8.2..8.2 | 1 | 5.236 | - | 8.2 | - | - (stored whole) | 5.236 (whole/whole (kernel not derivable)) | - | 4959.4 | 602.6 | loss_first_step=9.018663, loss_last_step=8.415815, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1127)`, ran on AMD Instinct MI325X (amd, DO) job a1127

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.5 ms) = 2.309; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.3 ms) = 2.583. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@a47bd9fb2) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | main board, one scored run | - | ok (main@a47bd9fb2 amd/a1127 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.3 | 1.3..1.3 | 1 | 2.583 | - | 1.3 | - | - (stored whole) | 2.583 (whole/whole (kernel not derivable)) | - | 2955.8 | 76.1 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.1 | 2.1..2.1 | 1 | 1.600 | - | 2.1 | - | - (stored whole) | 1.600 (whole/whole (kernel not derivable)) | - | 3007.0 | 76.1 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.9 | 1.9..1.9 | 1 | 1.813 | - | 1.9 | - | - (stored whole) | 1.813 (whole/whole (kernel not derivable)) | - | 4617.8 | 76.0 | loss_first_step=1.160498, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.5 | 1.5..1.5 | 1 | 2.309 | - | 1.5 | - | - (stored whole) | 2.309 (whole/whole (kernel not derivable)) | - | 4698.8 | 76.0 | loss_first_step=1.160367, loss_last_step=1.123438, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1189)`, ran on AMD Instinct MI325X (amd, DO) job a1189

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 5.7 ms) = 0.994; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 5.9 ms) = 0.953. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 5.6 | 5.6..5.6 | 1 | - | - | 5.6 | - | - (stored whole) | - | - | - | - | mean_nll=5.635910 | - | main board, one scored run | - | ok (main@9f83ea479 amd/a1189 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 40.2 | 40.2..40.2 | 1 | 0.140 | - | 40.2 | - | - (stored whole) | 0.140 (whole/whole (kernel not derivable)) | - | 3041.1 | 203.9 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.9 | 5.9..5.9 | 1 | 0.953 | - | 5.9 | - | - (stored whole) | 0.953 (whole/whole (kernel not derivable)) | - | 3698.7 | 173.9 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 40.9 | 40.9..40.9 | 1 | 0.138 | - | 40.9 | - | - (stored whole) | 0.138 (whole/whole (kernel not derivable)) | - | 4831.9 | 201.7 | mean_nll=5.635892 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.7 | 5.7..5.7 | 1 | 0.994 | - | 5.7 | - | - (stored whole) | 0.994 (whole/whole (kernel not derivable)) | - | 5545.2 | 156.2 | mean_nll=5.635953 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1189)`, ran on AMD Instinct MI325X (amd, DO) job a1189

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 127.6 ms) = 0.915; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 135.2 ms) = 0.863. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 116.7 | 116.7..116.7 | 1 | - | - | 116.7 | - | - (stored whole) | - | - | - | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | main board, one scored run | - | ok (main@9f83ea479 amd/a1189 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 135.2 | 135.2..135.2 | 1 | 0.863 | - | 135.2 | - | - (stored whole) | 0.863 (whole/whole (kernel not derivable)) | - | 3267.0 | 432.1 | loss_first_step=5.635909, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 127.6 | 127.6..127.6 | 1 | 0.915 | - | 127.6 | - | - (stored whole) | 0.915 (whole/whole (kernel not derivable)) | - | 4934.6 | 384.6 | loss_first_step=5.635892, loss_last_step=4.834043, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from opponents-20261006; measured this run) |

memory, ours, torch-compile-fp32, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1189)`, ran on AMD Instinct MI325X (amd, DO) job a1189

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.9 ms) = 1.661; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.6 ms) = 2.047. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@9f83ea479 amd/a1189 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.8 | 1.8..1.8 | 1 | 1.748 | - | 1.8 | - | - (stored whole) | 1.748 (whole/whole (kernel not derivable)) | - | 2873.1 | 147.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.6 | 1.6..1.6 | 1 | 2.047 | - | 1.6 | - | - (stored whole) | 2.047 (whole/whole (kernel not derivable)) | - | 2982.1 | 122.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.9 | 1.9..1.9 | 1 | 1.661 | - | 1.9 | - | - (stored whole) | 1.661 (whole/whole (kernel not derivable)) | - | 4144.3 | 139.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on cuda failed in round 0 (compile happens here): RuntimeError('self and mat2 must have the same dtype, but got Float and BFloat16')", "event": "error", "stage":) (copied from opponents-20261006; measured this run) |

memory, ours, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1194)`, ran on AMD Instinct MI325X (amd, DO) job a1194

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 7.5 ms) = 1.131. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1194 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 7.5 | 7.5..7.5 | 1 | 1.131 | - | - | 7.5 | - (stored kernel) | 1.131 (MIXED ours whole / arm kernel) | - | 2842.8 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 65.3 | 65.3..65.3 | 1 | 0.130 | - | - | 65.3 | - (stored kernel) | 0.130 (MIXED ours whole / arm kernel) | - | 2887.5 | 960.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adagrad / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 4.7 ms) = 0.696. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 3.3 | 3.3..3.3 | 1 | - | - | 3.3 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 4.7 | 4.7..4.7 | 1 | 0.696 | - | - | 4.7 | - (stored kernel) | 0.696 (MIXED ours whole / arm kernel) | - | 2831.4 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 282.7 | 282.7..282.7 | 1 | 0.012 | - | - | 282.7 | - (stored kernel) | 0.012 (MIXED ours whole / arm kernel) | - | 2923.9 | 832.0 | rel_fro_vs_torch_eager_fp32=1.603e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.7 ms) = 0.649. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 3.7 | 3.7..3.7 | 1 | - | - | 3.7 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.7 | 5.7..5.7 | 1 | 0.649 | - | - | 5.7 | - (stored kernel) | 0.649 (MIXED ours whole / arm kernel) | - | 2836.1 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 299.2 | 299.2..299.2 | 1 | 0.012 | - | - | 299.2 | - (stored kernel) | 0.012 (MIXED ours whole / arm kernel) | - | 2925.6 | 896.0 | rel_fro_vs_torch_eager_fp32=6.17e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 0.6 ms) = 0.773; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 0.669. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.7 | 0.7..0.7 | 1 | 0.669 | - | - | 0.7 | - (stored kernel) | 0.669 (MIXED ours whole / arm kernel) | - | 2665.7 | 224.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 0.484 | - | - | 1.0 | - (stored kernel) | 0.484 (MIXED ours whole / arm kernel) | - | 2839.4 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.8 | 0.8..0.8 | 1 | 0.595 | - | - | 0.8 | - (stored kernel) | 0.595 (MIXED ours whole / arm kernel) | - | 2670.0 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.6 | 0.6..0.6 | 1 | 0.773 | - | - | 0.6 | - (stored kernel) | 0.773 (MIXED ours whole / arm kernel) | - | 2778.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 0.9 ms) = 0.794; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.0 ms) = 0.746. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 0.7 | 0.7..0.7 | 1 | - | - | 0.7 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 0.746 | - | - | 1.0 | - (stored kernel) | 0.746 (MIXED ours whole / arm kernel) | - | 2686.0 | 541.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 0.744 | - | - | 1.0 | - (stored kernel) | 0.744 (MIXED ours whole / arm kernel) | - | 2862.2 | 540.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.9 | 0.9..0.9 | 1 | 0.788 | - | - | 0.9 | - (stored kernel) | 0.788 (MIXED ours whole / arm kernel) | - | 2685.8 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.9 | 0.9..0.9 | 1 | 0.794 | - | - | 0.9 | - (stored kernel) | 0.794 (MIXED ours whole / arm kernel) | - | 2794.9 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.9 ms) = 2.242; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.8 ms) = 2.478. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 2.1 | 2.1..2.1 | 1 | - | - | 2.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1212/work-batchnorm1d-synthetic-def/batchnorm1d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.8 | 0.8..0.8 | 1 | 2.478 | - | - | 0.8 | - (stored kernel) | 2.478 (MIXED ours whole / arm kernel) | - | 2988.4 | 320.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.2 | 1.2..1.2 | 1 | 1.762 | - | - | 1.2 | - (stored kernel) | 1.762 (MIXED ours whole / arm kernel) | - | 2895.2 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002683, rel_fro_vs_torch_eager_fp32=6.148e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.9 | 0.9..0.9 | 1 | 2.242 | - | - | 0.9 | - (stored kernel) | 2.242 (MIXED ours whole / arm kernel) | - | 2704.9 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.4 | 1.4..1.4 | 1 | 1.475 | - | - | 1.4 | - (stored kernel) | 1.475 (MIXED ours whole / arm kernel) | - | 2812.4 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002683, rel_fro_vs_torch_eager_fp32=6.148e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.7 ms) = 3.006; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.0 ms) = 2.237. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 2.2 | 2.2..2.2 | 1 | - | - | 2.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1200/work-batchnorm2d-synthetic-def/batchnorm2d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 2.237 | - | - | 1.0 | - (stored kernel) | 2.237 (MIXED ours whole / arm kernel) | - | 2962.8 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.2 | 1.2..1.2 | 1 | 1.864 | - | - | 1.2 | - (stored kernel) | 1.864 (MIXED ours whole / arm kernel) | - | 2868.7 | 246.0 | max_rel_diff_vs_torch_eager_fp32=0.001856, rel_fro_vs_torch_eager_fp32=7.706e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.7 | 0.7..0.7 | 1 | 3.006 | - | - | 0.7 | - (stored kernel) | 3.006 (MIXED ours whole / arm kernel) | - | 2692.6 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.4 | 1.4..1.4 | 1 | 1.602 | - | - | 1.4 | - (stored kernel) | 1.602 (MIXED ours whole / arm kernel) | - | 2798.0 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001856, rel_fro_vs_torch_eager_fp32=7.706e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 26.0 | 26.0..26.0 | 1 | - | - | 26.0 | - | - (stored whole) | - | - | - | - | accuracy=0.794050, logloss=5.350625 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 773.0 | 773.0..773.0 | 1 | 0.034 | - | 773.0 | 773.0 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 3874.7 | - | accuracy=0.794050, logloss=4.278741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 10.8 | 10.8..10.8 | 1 | - | - | 10.8 | - | - (stored whole) | - | - | - | - | accuracy=0.755560, logloss=0.557803 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 77.3 | 77.3..77.3 | 1 | 0.139 | - | 77.3 | 77.3 | 0.00 (cpu-arm) | 0.139 (whole/whole) | - | 446.1 | - | accuracy=0.755560, logloss=0.557802 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 18.0 | 18.0..18.0 | 1 | - | - | 18.0 | - | - (stored whole) | - | - | - | - | accuracy=0.838850, logloss=0.412625 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 89.3 | 89.3..89.3 | 1 | 0.201 | - | 89.3 | 89.3 | 0.00 (cpu-arm) | 0.201 (whole/whole) | - | 360.0 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 18.2 | 18.2..18.2 | 1 | - | - | 18.2 | - | - (stored whole) | - | - | - | - | accuracy=0.765850, logloss=0.538866 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 74.9 | 74.9..74.9 | 1 | 0.243 | - | 74.9 | 74.9 | 0.00 (cpu-arm) | 0.243 (whole/whole) | - | 323.7 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on AMD Instinct MI325X (amd, DO) job a0866

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@bb7cc2fa0) | identical | 163.4 | 163.4..163.4 | 1 | - | - | 163.4 | - | - (stored whole) | - | - | - | - | relative_residual=2.9e-07 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 20.1 | 20.1..20.1 | 1 | 8.146 | - | - | 20.1 | - (stored kernel) | 8.146 (MIXED ours whole / arm kernel) | - | 3745.3 | 768.0 | relative_residual=1.044e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 7138.9 | 7138.9..7138.9 | 1 | 0.023 | - | 7138.9 | 7138.9 | 0.00 (cpu-arm) | 0.023 (whole/whole) | - | 2650.2 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 251.7 ms) = 1.248; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 172.9 ms) = 1.817. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 314.2 | 314.2..314.2 | 1 | - | - | 314.2 | - | - (stored whole) | - | - | - | - | accuracy=1.000000 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 172.9 | 172.9..172.9 | 1 | 1.817 | - | - | 172.9 | - (stored kernel) | 1.817 (MIXED ours whole / arm kernel) | - | 3768.4 | 271.5 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 245.0 | 245.0..245.0 | 1 | 1.282 | - | - | 245.0 | - (stored kernel) | 1.282 (MIXED ours whole / arm kernel) | - | 3778.8 | 198.0 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 251.7 | 251.7..251.7 | 1 | 1.248 | - | - | 251.7 | - (stored kernel) | 1.248 (MIXED ours whole / arm kernel) | - | 5492.3 | 278.6 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 269.9 | 269.9..269.9 | 1 | 1.164 | - | - | 269.9 | - (stored kernel) | 1.164 (MIXED ours whole / arm kernel) | - | 5267.5 | 277.6 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 67.3 | 67.3..67.3 | 1 | - | - | 67.3 | - | - (stored whole) | - | - | - | - | accuracy=0.849360, logloss=3.762524 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 196.8 | 196.8..196.8 | 1 | 0.342 | - | 196.8 | 196.8 | 0.00 (cpu-arm) | 0.342 (whole/whole) | - | 3958.9 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 27.6 | 27.6..27.6 | 1 | - | - | 27.6 | - | - (stored whole) | - | - | - | - | accuracy=0.678020, logloss=0.715492 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38.2 | 38.2..38.2 | 1 | 0.723 | - | 38.2 | 38.2 | 0.00 (cpu-arm) | 0.723 (whole/whole) | - | 450.3 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0879)`, ran on AMD Instinct MI325X (amd, DO) job a0879

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 21.0 | 21.0..21.0 | 1 | - | - | 21.0 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559491 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0879 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 269.7 | 269.7..269.7 | 1 | 0.078 | - | 269.7 | 269.7 | 0.00 (cpu-arm) | 0.078 (whole/whole) | - | 4455.8 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 0.7 | 0.7..0.7 | 1 | - | - | 0.7 | - | - (stored whole) | - | - | - | - | n_components=81 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 5.7 | 5.7..5.7 | 1 | 0.118 | - | 5.7 | 5.7 | 0.00 (cpu-arm) | 0.118 (whole/whole) | - | 117.4 | - | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | - | - | n_components=588 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 5.3 | 5.3..5.3 | 1 | 0.122 | - | 5.3 | 5.3 | 0.00 (cpu-arm) | 0.122 (whole/whole) | - | 99.9 | - | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.3 ms) = 5.777; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.0 ms) = 3.602. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 7.3 | 7.3..7.3 | 1 | - | - | 7.3 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1212/work-conv1d-synthetic-def/conv1d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.0 | 2.0..2.0 | 1 | 3.602 | - | - | 2.0 | - (stored kernel) | 3.602 (MIXED ours whole / arm kernel) | - | 3393.2 | 896.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.2 | 2.2..2.2 | 1 | 3.312 | - | - | 2.2 | - (stored kernel) | 3.312 (MIXED ours whole / arm kernel) | - | 3085.1 | 896.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.3 | 1.3..1.3 | 1 | 5.777 | - | - | 1.3 | - (stored kernel) | 5.777 (MIXED ours whole / arm kernel) | - | 3529.4 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.2 | 2.2..2.2 | 1 | 3.377 | - | - | 2.2 | - (stored kernel) | 3.377 (MIXED ours whole / arm kernel) | - | 3097.4 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.1 ms) = 5.747; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.1 ms) = 5.629. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 6.0 | 6.0..6.0 | 1 | - | - | 6.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1200/work-conv2d-synthetic-def/conv2d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.1 | 1.1..1.1 | 1 | 5.629 | - | - | 1.1 | - (stored kernel) | 5.629 (MIXED ours whole / arm kernel) | - | 3154.2 | 345.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.6 | 1.6..1.6 | 1 | 3.874 | - | - | 1.6 | - (stored kernel) | 3.874 (MIXED ours whole / arm kernel) | - | 3062.4 | 345.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.1 | 1.1..1.1 | 1 | 5.747 | - | - | 1.1 | - (stored kernel) | 5.747 (MIXED ours whole / arm kernel) | - | 3160.3 | 320.8 | max_rel_diff_vs_torch_eager_fp32=3417.825326, rel_fro_vs_torch_eager_fp32=0.004438 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.7 | 1.7..1.7 | 1 | 3.533 | - | - | 1.7 | - (stored kernel) | 3.533 (MIXED ours whole / arm kernel) | - | 3019.1 | 319.8 | max_rel_diff_vs_torch_eager_fp32=3417.825326, rel_fro_vs_torch_eager_fp32=0.004438 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on AMD Instinct MI325X (amd, DO) job a1069

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 1679.7 | 1679.7..1679.7 | 1 | - | - | 1679.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=13.931153 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 94.5 | 94.5..94.5 | 1 | 17.782 | - | 94.5 | 94.5 | 0.00 (cpu-arm) | 17.782 (whole/whole) | - | 58.1 | - | forecast_rmse=26.588738 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 93.0 | 93.0..93.0 | 1 | 18.056 | - | 93.0 | 93.0 | 0.00 (cpu-arm) | 18.056 (whole/whole) | - | 283.5 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on AMD Instinct MI325X (amd, DO) job a1069

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 1770.6 | 1770.6..1770.6 | 1 | - | - | 1770.6 | - | - (stored whole) | - | - | - | - | forecast_rmse=96.690449 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 94.0 | 94.0..94.0 | 1 | 18.828 | - | 94.0 | 94.0 | 0.00 (cpu-arm) | 18.828 (whole/whole) | - | 57.9 | - | forecast_rmse=196.955273 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 63.3 | 63.3..63.3 | 1 | 27.968 | - | 63.3 | 63.3 | 0.00 (cpu-arm) | 27.968 (whole/whole) | - | 283.6 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 0.506. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 0.4 | 0.4..0.4 | 1 | - | - | 0.4 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.7 | 0.7..0.7 | 1 | 0.506 | - | - | 0.7 | - (stored kernel) | 0.506 (MIXED ours whole / arm kernel) | - | 2641.0 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.8 | 0.8..0.8 | 1 | 0.440 | - | - | 0.8 | - (stored kernel) | 0.440 (MIXED ours whole / arm kernel) | - | 2820.8 | 246.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on AMD Instinct MI325X (amd, DO) job a0866

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@bb7cc2fa0) | identical | 8620.1 | 8620.1..8620.1 | 1 | - | - | 8620.1 | - | - (stored whole) | - | - | - | - | max_eigenvalue_error=5.95e-05, relative_residual=5.251e-05 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1922.1 | 1922.1..1922.1 | 1 | 4.485 | - | - | 1922.1 | - (stored kernel) | 4.485 (MIXED ours whole / arm kernel) | - | 2900.1 | 1220.6 | max_eigenvalue_error=1.254e-06, relative_residual=1.269e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 2356.5 | 2356.5..2356.5 | 1 | 3.658 | - | 2356.5 | 2356.5 | 0.00 (cpu-arm) | 3.658 (whole/whole) | - | 970.0 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### embedding / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.0 ms) = 2.094. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 2.0 | 2.0..2.0 | 1 | - | - | 2.0 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.1 | 1.1..1.1 | 1 | 1.757 | - | - | 1.1 | - (stored kernel) | 1.757 (MIXED ours whole / arm kernel) | - | 2811.2 | 782.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 2.094 | - | - | 1.0 | - (stored kernel) | 2.094 (MIXED ours whole / arm kernel) | - | 2941.9 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1554)`, ran on AMD Instinct MI325X (amd, DO) job a1554

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 835.5 | 835.5..835.5 | 1 | - | - | 835.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.326805, rmse=0.685379 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1554 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 47198.6 | 47198.6..47198.6 | 1 | 0.018 | - | 47198.6 | 47198.6 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 8285.3 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1554)`, ran on AMD Instinct MI325X (amd, DO) job a1554

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 257.0 | 257.0..257.0 | 1 | - | - | 257.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909004, rmse=4.804486 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1554 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 279.5 | 279.5..279.5 | 1 | 0.919 | - | 279.5 | 279.5 | 0.00 (cpu-arm) | 0.919 (whole/whole) | - | 673.5 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1130)`, ran on AMD Instinct MI325X (amd, DO) job a1130

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@a47bd9fb2) | identical | 28.2 | 28.2..28.2 | 1 | - | - | 28.2 | - | - (stored whole) | - | - | - | - | accuracy=0.876570, logloss=3.574420 | - | main board, one scored run | - | ok (main@a47bd9fb2 amd/a1130 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 420.3 | 420.3..420.3 | 1 | 0.067 | - | 420.3 | 420.3 | 0.00 (cpu-arm) | 0.067 (whole/whole) | - | 2633.4 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1130)`, ran on AMD Instinct MI325X (amd, DO) job a1130

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@a47bd9fb2) | identical | 12.2 | 12.2..12.2 | 1 | - | - | 12.2 | - | - (stored whole) | - | - | - | - | accuracy=0.719820, logloss=1.132249 | - | main board, one scored run | - | ok (main@a47bd9fb2 amd/a1130 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 82.1 | 82.1..82.1 | 1 | 0.149 | - | 82.1 | 82.1 | 0.00 (cpu-arm) | 0.149 (whole/whole) | - | 324.2 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.3 ms) = 0.921; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.6 ms) = 0.834. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-gcn-istella-def/gcn-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 16.9 | 16.9..16.9 | 1 | 0.180 | - | - | 16.9 | - (stored kernel) | 0.180 (MIXED ours whole / arm kernel) | - | 3417.9 | 1918.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.6 | 3.6..3.6 | 1 | 0.834 | - | - | 3.6 | - (stored kernel) | 0.834 (MIXED ours whole / arm kernel) | - | 3477.3 | 383.2 | max_rel_diff_vs_torch_eager_fp32=0.025557, rel_fro_vs_torch_eager_fp32=9.36e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 10.7 | 10.7..10.7 | 1 | 0.285 | - | - | 10.7 | - (stored kernel) | 0.285 (MIXED ours whole / arm kernel) | - | 4528.1 | 2307.5 | max_rel_diff_vs_torch_eager_fp32=2662.412698, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.3 | 3.3..3.3 | 1 | 0.921 | - | - | 3.3 | - (stored kernel) | 0.921 (MIXED ours whole / arm kernel) | - | 5066.5 | 766.2 | max_rel_diff_vs_torch_eager_fp32=2662.401917, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 2.8 ms) = 0.491; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.2 ms) = 0.428. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 1.4 | 1.4..1.4 | 1 | - | - | 1.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-gcn-taxi-def/gcn-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 14.1 | 14.1..14.1 | 1 | 0.097 | - | - | 14.1 | - (stored kernel) | 0.097 (MIXED ours whole / arm kernel) | - | 3333.2 | 1573.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.2 | 3.2..3.2 | 1 | 0.428 | - | - | 3.2 | - (stored kernel) | 0.428 (MIXED ours whole / arm kernel) | - | 3461.6 | 295.5 | max_rel_diff_vs_torch_eager_fp32=0.001702, rel_fro_vs_torch_eager_fp32=7.846e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 9.0 | 9.0..9.0 | 1 | 0.152 | - | - | 9.0 | - (stored kernel) | 0.152 (MIXED ours whole / arm kernel) | - | 4436.3 | 1858.7 | max_rel_diff_vs_torch_eager_fp32=1509.509399, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.8 | 2.8..2.8 | 1 | 0.491 | - | - | 2.8 | - (stored kernel) | 0.491 (MIXED ours whole / arm kernel) | - | 4978.9 | 574.3 | max_rel_diff_vs_torch_eager_fp32=1509.509515, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.4 ms) = 2.661; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.5 ms) = 2.309. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.2 | 1.2..1.2 | 1 | - | - | 1.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1212/work-global-avgpool-synthetic-def/global-avgpool-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.5 | 0.5..0.5 | 1 | 2.309 | - | - | 0.5 | - (stored kernel) | 2.309 (MIXED ours whole / arm kernel) | - | 2644.2 | 12.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.9 | 0.9..0.9 | 1 | 1.330 | - | - | 0.9 | - (stored kernel) | 1.330 (MIXED ours whole / arm kernel) | - | 2813.0 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.014597, rel_fro_vs_torch_eager_fp32=9.508e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.4 | 0.4..0.4 | 1 | 2.661 | - | - | 0.4 | - (stored kernel) | 2.661 (MIXED ours whole / arm kernel) | - | 2644.3 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.5 | 0.5..0.5 | 1 | 2.267 | - | - | 0.5 | - (stored kernel) | 2.267 (MIXED ours whole / arm kernel) | - | 2745.8 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.014597, rel_fro_vs_torch_eager_fp32=9.508e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.6 ms) = 1.934; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.5 ms) = 2.535. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.2 | 1.2..1.2 | 1 | - | - | 1.2 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.5 | 0.5..0.5 | 1 | 2.535 | - | - | 0.5 | - (stored kernel) | 2.535 (MIXED ours whole / arm kernel) | - | 2639.6 | 12.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 1.225 | - | - | 1.0 | - (stored kernel) | 1.225 (MIXED ours whole / arm kernel) | - | 2748.0 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.6 | 0.6..0.6 | 1 | 1.934 | - | - | 0.6 | - (stored kernel) | 1.934 (MIXED ours whole / arm kernel) | - | 2639.6 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.7 | 0.7..0.7 | 1 | 1.853 | - | - | 0.7 | - (stored kernel) | 1.853 (MIXED ours whole / arm kernel) | - | 2746.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.2 ms) = 1.843; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.4 ms) = 1.756. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 5.9 | 5.9..5.9 | 1 | - | - | 5.9 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-graphsage-istella-def/graphsage-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 12.5 | 12.5..12.5 | 1 | 0.474 | - | - | 12.5 | - (stored kernel) | 0.474 (MIXED ours whole / arm kernel) | - | 3373.3 | 1651.2 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.4 | 3.4..3.4 | 1 | 1.756 | - | - | 3.4 | - (stored kernel) | 1.756 (MIXED ours whole / arm kernel) | - | 3458.5 | 463.5 | max_rel_diff_vs_torch_eager_fp32=0.140845, rel_fro_vs_torch_eager_fp32=8.05e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 12.6 | 12.6..12.6 | 1 | 0.469 | - | - | 12.6 | - (stored kernel) | 0.469 (MIXED ours whole / arm kernel) | - | 4489.7 | 1626.7 | max_rel_diff_vs_torch_eager_fp32=3907.144070, rel_fro_vs_torch_eager_fp32=0.003366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.2 | 3.2..3.2 | 1 | 1.843 | - | - | 3.2 | - (stored kernel) | 1.843 (MIXED ours whole / arm kernel) | - | 5119.7 | 433.0 | max_rel_diff_vs_torch_eager_fp32=3678.232431, rel_fro_vs_torch_eager_fp32=0.003054 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.4 ms) = 1.190; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.4 ms) = 1.157. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-graphsage-taxi-def/graphsage-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.4 | 1.4..1.4 | 1 | 1.157 | - | - | 1.4 | - (stored kernel) | 1.157 (MIXED ours whole / arm kernel) | - | 3308.4 | 299.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.5 | 1.5..1.5 | 1 | 1.086 | - | - | 1.5 | - (stored kernel) | 1.086 (MIXED ours whole / arm kernel) | - | 3439.0 | 299.7 | max_rel_diff_vs_torch_eager_fp32=0.162162, rel_fro_vs_torch_eager_fp32=8.776e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.5 | 1.5..1.5 | 1 | 1.124 | - | - | 1.5 | - (stored kernel) | 1.124 (MIXED ours whole / arm kernel) | - | 4422.1 | 227.1 | max_rel_diff_vs_torch_eager_fp32=4680.142857, rel_fro_vs_torch_eager_fp32=0.003557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.4 | 1.4..1.4 | 1 | 1.190 | - | - | 1.4 | - (stored kernel) | 1.190 (MIXED ours whole / arm kernel) | - | 5037.8 | 230.2 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1852.7 ms) = 0.278; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1946.7 ms) = 0.265. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 514.9 | 514.9..514.9 | 1 | - | - | 514.9 | - | - (stored whole) | - | - | - | - | accuracy=0.971625, logloss=0.070054 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1946.7 | 1946.7..1946.7 | 1 | 0.265 | - | - | 1946.7 | - (stored kernel) | 0.265 (MIXED ours whole / arm kernel) | - | 3217.3 | 324.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1997.8 | 1997.8..1997.8 | 1 | 0.258 | - | - | 1997.8 | - (stored kernel) | 0.258 (MIXED ours whole / arm kernel) | - | 3248.0 | 324.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1852.7 | 1852.7..1852.7 | 1 | 0.278 | - | - | 1852.7 | - (stored kernel) | 0.278 (MIXED ours whole / arm kernel) | - | 4957.7 | 168.7 | accuracy=0.971788 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1973.8 | 1973.8..1973.8 | 1 | 0.261 | - | - | 1973.8 | - (stored kernel) | 0.261 (MIXED ours whole / arm kernel) | - | 5008.5 | 168.7 | accuracy=0.971788 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1899.7 ms) = 0.271; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1792.8 ms) = 0.288. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 515.5 | 515.5..515.5 | 1 | - | - | 515.5 | - | - (stored whole) | - | - | - | - | accuracy=0.865723, logloss=0.303537 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1792.8 | 1792.8..1792.8 | 1 | 0.288 | - | - | 1792.8 | - (stored kernel) | 0.288 (MIXED ours whole / arm kernel) | - | 3610.7 | 324.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2027.5 | 2027.5..2027.5 | 1 | 0.254 | - | - | 2027.5 | - (stored kernel) | 0.254 (MIXED ours whole / arm kernel) | - | 3258.5 | 324.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2025.2 | 2025.2..2025.2 | 1 | 0.255 | - | - | 2025.2 | - (stored kernel) | 0.255 (MIXED ours whole / arm kernel) | - | 5294.2 | 168.7 | accuracy=0.865777 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1899.7 | 1899.7..1899.7 | 1 | 0.271 | - | - | 1899.7 | - (stored kernel) | 0.271 (MIXED ours whole / arm kernel) | - | 5008.7 | 168.7 | accuracy=0.865777 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2078.2 ms) = 0.247; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1993.7 ms) = 0.257. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 513.3 | 513.3..513.3 | 1 | - | - | 513.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.982337, rmse=0.153975 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1993.7 | 1993.7..1993.7 | 1 | 0.257 | - | - | 1993.7 | - (stored kernel) | 0.257 (MIXED ours whole / arm kernel) | - | 3044.7 | 324.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1996.8 | 1996.8..1996.8 | 1 | 0.257 | - | - | 1996.8 | - (stored kernel) | 0.257 (MIXED ours whole / arm kernel) | - | 3072.8 | 324.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2078.2 | 2078.2..2078.2 | 1 | 0.247 | - | - | 2078.2 | - (stored kernel) | 0.247 (MIXED ours whole / arm kernel) | - | 3656.3 | 168.4 | finite=True, r2=0.981940, rmse=0.155700 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2123.0 | 2123.0..2123.0 | 1 | 0.242 | - | - | 2123.0 | - (stored kernel) | 0.242 (MIXED ours whole / arm kernel) | - | 3708.5 | 168.4 | finite=True, r2=0.981940, rmse=0.155700 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1801.3 ms) = 0.285; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 2006.3 ms) = 0.256. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 513.4 | 513.4..513.4 | 1 | - | - | 513.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.747875, rmse=0.544554 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2012.1 | 2012.1..2012.1 | 1 | 0.255 | - | - | 2012.1 | - (stored kernel) | 0.255 (MIXED ours whole / arm kernel) | - | 3044.6 | 324.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2006.3 | 2006.3..2006.3 | 1 | 0.256 | - | - | 2006.3 | - (stored kernel) | 0.256 (MIXED ours whole / arm kernel) | - | 3072.3 | 324.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1801.3 | 1801.3..1801.3 | 1 | 0.285 | - | - | 1801.3 | - (stored kernel) | 0.285 (MIXED ours whole / arm kernel) | - | 3655.7 | 168.4 | finite=True, r2=0.748254, rmse=0.544144 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2127.7 | 2127.7..2127.7 | 1 | 0.241 | - | - | 2127.7 | - (stored kernel) | 0.241 (MIXED ours whole / arm kernel) | - | 3708.4 | 168.4 | finite=True, r2=0.748254, rmse=0.544144 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1160)`, ran on AMD Instinct MI325X (amd, DO) job a1160

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 541.5 | 541.5..541.5 | 1 | - | - | 541.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.802100 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1160 2026-10-09; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 5118.6 | 5118.6..5118.6 | 1 | 0.106 | - | 5118.6 | 5118.6 | 0.00 (cpu-arm) | 0.106 (whole/whole) | - | 754.2 | - | recall_at_10=0.801550 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1160)`, ran on AMD Instinct MI325X (amd, DO) job a1160

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 84.6 | 84.6..84.6 | 1 | - | - | 84.6 | - | - (stored whole) | - | - | - | - | recall_at_10=0.981250 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1160 2026-10-09; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 914.0 | 914.0..914.0 | 1 | 0.093 | - | 914.0 | 914.0 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 142.7 | - | recall_at_10=0.979450 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on AMD Instinct MI325X (amd, DO) job a0904

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 8230.1 | 8230.1..8230.1 | 1 | - | - | 8230.1 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 10339.5 | 10339.5..10339.5 | 1 | 0.796 | - | 10339.5 | 10339.5 | 0.00 (cpu-arm) | 0.796 (whole/whole) | - | 2196.3 | - | rel_error_vs_exact=8.63e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on AMD Instinct MI325X (amd, DO) job a0904

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 252.6 | 252.6..252.6 | 1 | - | - | 252.6 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.056e-07 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 1154.2 | 1154.2..1154.2 | 1 | 0.219 | - | 1154.2 | 1154.2 | 0.00 (cpu-arm) | 0.219 (whole/whole) | - | 405.9 | - | rel_error_vs_exact=8.149e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1160)`, ran on AMD Instinct MI325X (amd, DO) job a1160

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 25.2 | 25.2..25.2 | 1 | - | - | 25.2 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1160 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 16.2 | 16.2..16.2 | 1 | 1.555 | - | 16.2 | 16.2 | 0.00 (cpu-arm) | 1.555 (whole/whole) | - | 3528.4 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1160)`, ran on AMD Instinct MI325X (amd, DO) job a1160

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | masked_rmse=6.151696 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1160 2026-10-09; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1.9 | 1.9..1.9 | 1 | 0.579 | - | 1.9 | 1.9 | 0.00 (cpu-arm) | 0.579 (whole/whole) | - | 1966.8 | - | masked_rmse=5.263919 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lamb / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on AMD Instinct MI325X (amd, DO) job a0873

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 13.4 | 13.4..13.4 | 1 | - | - | 13.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-lamb-synthetic-def/lamb-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1554)`, ran on AMD Instinct MI325X (amd, DO) job a1554

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 793.0 | 793.0..793.0 | 1 | - | - | 793.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.325506, rmse=0.686040 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1554 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 42194.4 | 42194.4..42194.4 | 1 | 0.019 | - | 42194.4 | 42194.4 | 0.00 (cpu-arm) | 0.019 (whole/whole) | - | 8912.8 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1554)`, ran on AMD Instinct MI325X (amd, DO) job a1554

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 203.8 | 203.8..203.8 | 1 | - | - | 203.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909038, rmse=4.803593 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1554 2026-10-10; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 277.2 | 277.2..277.2 | 1 | 0.735 | - | 277.2 | 277.2 | 0.00 (cpu-arm) | 0.735 (whole/whole) | - | 680.5 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.9 ms) = 1.319; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.8 ms) = 1.546. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 1.2 | 1.2..1.2 | 1 | - | - | 1.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1200/work-layernorm-synthetic-def/layernorm-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.8 | 0.8..0.8 | 1 | 1.546 | - | - | 0.8 | - (stored kernel) | 1.546 (MIXED ours whole / arm kernel) | - | 2690.0 | 320.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 1.189 | - | - | 1.0 | - (stored kernel) | 1.189 (MIXED ours whole / arm kernel) | - | 2866.8 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.006116, rel_fro_vs_torch_eager_fp32=5.286e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.9 | 0.9..0.9 | 1 | 1.319 | - | - | 0.9 | - (stored kernel) | 1.319 (MIXED ours whole / arm kernel) | - | 2687.3 | 320.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.2 | 1.2..1.2 | 1 | 1.013 | - | - | 1.2 | - (stored kernel) | 1.013 (MIXED ours whole / arm kernel) | - | 2798.9 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.006116, rel_fro_vs_torch_eager_fp32=5.286e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'elementwise_affine': True, 'eps': 1e-05, 'normalized_shape': 1024}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on AMD Instinct MI325X (amd, DO) job a1066

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 315.8 | 315.8..315.8 | 1 | - | - | 315.8 | - | - (stored whole) | - | - | - | - | accuracy=0.912830, logloss=0.235875 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3925.4 | 3925.4..3925.4 | 1 | 0.080 | - | 3925.4 | 3925.4 | 0.00 (cpu-arm) | 0.080 (whole/whole) | - | 5444.6 | - | accuracy=0.899520, logloss=0.439049 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on AMD Instinct MI325X (amd, DO) job a1066

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 17.4 | 17.4..17.4 | 1 | - | - | 17.4 | - | - (stored whole) | - | - | - | - | accuracy=0.762530, logloss=0.539763 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 230.9 | 230.9..230.9 | 1 | 0.075 | - | 230.9 | 230.9 | 0.00 (cpu-arm) | 0.075 (whole/whole) | - | 513.8 | - | accuracy=0.762530, logloss=0.539767 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lion / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on AMD Instinct MI325X (amd, DO) job a0873

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-lion-synthetic-def/lion-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'betas': [0.9, 0.99], 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 359.9 | 359.9..359.9 | 1 | - | - | 359.9 | - | - (stored whole) | - | - | - | - | modularity=0.911187, n_communities=40 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 1756.7 | 1756.7..1756.7 | 1 | 0.205 | - | 1756.7 | 1756.7 | 0.00 (cpu-arm) | 0.205 (whole/whole) | - | 238.5 | - | modularity=0.908460, n_communities=40 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 71.2 | 71.2..71.2 | 1 | - | - | 71.2 | - | - (stored whole) | - | - | - | - | modularity=0.941953, n_communities=58 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 1078.3 | 1078.3..1078.3 | 1 | 0.066 | - | 1078.3 | 1078.3 | 0.00 (cpu-arm) | 0.066 (whole/whole) | - | 203.7 | - | modularity=0.940781, n_communities=56 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 854.9 ms) = 0.739; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 894.9 ms) = 0.706. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 631.6 | 631.6..631.6 | 1 | - | - | 631.6 | - | - (stored whole) | - | - | - | - | accuracy=0.967068, logloss=0.079720 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 903.2 | 903.2..903.2 | 1 | 0.699 | - | - | 903.2 | - (stored kernel) | 0.699 (MIXED ours whole / arm kernel) | - | 3208.6 | 350.8 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 894.9 | 894.9..894.9 | 1 | 0.706 | - | - | 894.9 | - (stored kernel) | 0.706 (MIXED ours whole / arm kernel) | - | 3257.9 | 350.8 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 854.9 | 854.9..854.9 | 1 | 0.739 | - | - | 854.9 | - (stored kernel) | 0.739 (MIXED ours whole / arm kernel) | - | 4967.1 | 182.8 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1044.0 | 1044.0..1044.0 | 1 | 0.605 | - | - | 1044.0 | - (stored kernel) | 0.605 (MIXED ours whole / arm kernel) | - | 5000.0 | 182.8 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 977.6 ms) = 0.645; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 815.7 ms) = 0.773. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 630.7 | 630.7..630.7 | 1 | - | - | 630.7 | - | - (stored whole) | - | - | - | - | accuracy=0.870443, logloss=0.297146 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 861.1 | 861.1..861.1 | 1 | 0.732 | - | - | 861.1 | - (stored kernel) | 0.732 (MIXED ours whole / arm kernel) | - | 3603.8 | 350.8 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 815.7 | 815.7..815.7 | 1 | 0.773 | - | - | 815.7 | - (stored kernel) | 0.773 (MIXED ours whole / arm kernel) | - | 3258.3 | 350.8 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1039.6 | 1039.6..1039.6 | 1 | 0.607 | - | - | 1039.6 | - (stored kernel) | 0.607 (MIXED ours whole / arm kernel) | - | 5292.2 | 182.8 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 977.6 | 977.6..977.6 | 1 | 0.645 | - | - | 977.6 | - (stored kernel) | 0.645 (MIXED ours whole / arm kernel) | - | 5008.3 | 182.8 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1053.4 ms) = 0.597; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 923.1 ms) = 0.681. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 628.6 | 628.6..628.6 | 1 | - | - | 628.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.979398, rmse=0.166295 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 923.1 | 923.1..923.1 | 1 | 0.681 | - | - | 923.1 | - (stored kernel) | 0.681 (MIXED ours whole / arm kernel) | - | 3036.9 | 350.5 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 973.6 | 973.6..973.6 | 1 | 0.646 | - | - | 973.6 | - (stored kernel) | 0.646 (MIXED ours whole / arm kernel) | - | 3082.7 | 350.5 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1053.4 | 1053.4..1053.4 | 1 | 0.597 | - | - | 1053.4 | - (stored kernel) | 0.597 (MIXED ours whole / arm kernel) | - | 3660.4 | 182.5 | finite=True, r2=0.981004, rmse=0.159679 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1065.8 | 1065.8..1065.8 | 1 | 0.590 | - | - | 1065.8 | - (stored kernel) | 0.590 (MIXED ours whole / arm kernel) | - | 3706.1 | 182.5 | finite=True, r2=0.981004, rmse=0.159679 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1017.5 ms) = 0.617; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 920.3 ms) = 0.682. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 627.9 | 627.9..627.9 | 1 | - | - | 627.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.754306, rmse=0.537563 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 920.3 | 920.3..920.3 | 1 | 0.682 | - | - | 920.3 | - (stored kernel) | 0.682 (MIXED ours whole / arm kernel) | - | 3043.3 | 350.5 | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 976.5 | 976.5..976.5 | 1 | 0.643 | - | - | 976.5 | - (stored kernel) | 0.643 (MIXED ours whole / arm kernel) | - | 3084.4 | 350.5 | finite=True, r2=0.751679, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1017.5 | 1017.5..1017.5 | 1 | 0.617 | - | - | 1017.5 | - (stored kernel) | 0.617 (MIXED ours whole / arm kernel) | - | 3667.0 | 182.5 | finite=True, r2=0.751591, rmse=0.540526 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1057.7 | 1057.7..1057.7 | 1 | 0.594 | - | - | 1057.7 | - (stored kernel) | 0.594 (MIXED ours whole / arm kernel) | - | 3707.8 | 182.5 | finite=True, r2=0.751591, rmse=0.540526 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on AMD Instinct MI325X (amd, DO) job a0868

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 288.9 | 288.9..288.9 | 1 | - | - | 288.9 | - | - (stored whole) | - | - | - | - | relative_residual=0.849956 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1138.9 | 1138.9..1138.9 | 1 | 0.254 | - | - | 1138.9 | - (stored kernel) | 0.254 (MIXED ours whole / arm kernel) | - | 4308.1 | 1691.4 | relative_residual=nan | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 2526.8 | 2526.8..2526.8 | 1 | 0.114 | - | 2526.8 | 2526.8 | 0.00 (cpu-arm) | 0.114 (whole/whole) | - | 4351.2 | - | relative_residual=0.876106 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on AMD Instinct MI325X (amd, DO) job a0868

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 7.7 | 7.7..7.7 | 1 | - | - | 7.7 | - | - (stored whole) | - | - | - | - | relative_residual=0.756366 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 33.3 | 33.3..33.3 | 1 | 0.231 | - | - | 33.3 | - (stored kernel) | 0.231 (MIXED ours whole / arm kernel) | - | 3132.5 | 95.4 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 91.5 | 91.5..91.5 | 1 | 0.084 | - | 91.5 | 91.5 | 0.00 (cpu-arm) | 0.084 (whole/whole) | - | 284.0 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on AMD Instinct MI325X (amd, DO) job a0866

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@bb7cc2fa0) | identical | 574.6 | 574.6..574.6 | 1 | - | - | 574.6 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 86.9 | 86.9..86.9 | 1 | 6.613 | - | - | 86.9 | - (stored kernel) | 6.613 (MIXED ours whole / arm kernel) | - | 2673.1 | 710.0 | relative_residual=4.003e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| scipy-cpu | scipy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scipy 1.18.1 | opponent | 4553.9 | 4553.9..4553.9 | 1 | 0.126 | - | 4553.9 | 4553.9 | 0.00 (cpu-arm) | 0.126 (whole/whole) | - | 628.0 | - | relative_residual=3.439e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, scipy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on AMD Instinct MI325X (amd, DO) job a0866

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@bb7cc2fa0) | identical | 324.2 | 324.2..324.2 | 1 | - | - | 324.2 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 86.3 | 86.3..86.3 | 1 | 3.755 | - | - | 86.3 | - (stored kernel) | 3.755 (MIXED ours whole / arm kernel) | - | 2694.0 | 712.0 | relative_residual=4.041e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 5413.8 | 5413.8..5413.8 | 1 | 0.060 | - | 5413.8 | 5413.8 | 0.00 (cpu-arm) | 0.060 (whole/whole) | - | 1389.2 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.8 ms) = 0.572; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 0.8 ms) = 0.525. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 0.4 | 0.4..0.4 | 1 | - | - | 0.4 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.9 | 0.9..0.9 | 1 | 0.505 | - | - | 0.9 | - (stored kernel) | 0.505 (MIXED ours whole / arm kernel) | - | 2672.8 | 288.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.8 | 0.8..0.8 | 1 | 0.525 | - | - | 0.8 | - (stored kernel) | 0.525 (MIXED ours whole / arm kernel) | - | 2846.6 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.8 | 0.8..0.8 | 1 | 0.572 | - | - | 0.8 | - (stored kernel) | 0.572 (MIXED ours whole / arm kernel) | - | 2673.1 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.8 | 0.8..0.8 | 1 | 0.530 | - | - | 0.8 | - (stored kernel) | 0.530 (MIXED ours whole / arm kernel) | - | 2779.4 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.0 ms) = 0.659; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.9 ms) = 0.686. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 0.9 | 0.9..0.9 | 1 | 0.686 | - | - | 0.9 | - (stored kernel) | 0.686 (MIXED ours whole / arm kernel) | - | 2688.5 | 639.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 0.656 | - | - | 1.0 | - (stored kernel) | 0.656 (MIXED ours whole / arm kernel) | - | 2864.7 | 552.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.0 | 1.0..1.0 | 1 | 0.659 | - | - | 1.0 | - (stored kernel) | 0.659 (MIXED ours whole / arm kernel) | - | 2688.5 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1.1 | 1.1..1.1 | 1 | 0.582 | - | - | 1.1 | - (stored kernel) | 0.582 (MIXED ours whole / arm kernel) | - | 2795.8 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### moe / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.2 ms) = 4.731; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.6 ms) = 2.709. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 15.2 | 15.2..15.2 | 1 | - | - | 15.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1200/work-moe-synthetic-def/moe-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.6 | 5.6..5.6 | 1 | 2.709 | - | - | 5.6 | - (stored kernel) | 2.709 (MIXED ours whole / arm kernel) | - | 3220.1 | 514.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 6.8 | 6.8..6.8 | 1 | 2.222 | - | - | 6.8 | - (stored kernel) | 2.222 (MIXED ours whole / arm kernel) | - | 3395.8 | 514.6 | max_rel_diff_vs_torch_eager_fp32=0.044703, rel_fro_vs_torch_eager_fp32=2.011e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.6 | 3.6..3.6 | 1 | 4.252 | - | - | 3.6 | - (stored kernel) | 4.252 (MIXED ours whole / arm kernel) | - | 3597.0 | 623.0 | max_rel_diff_vs_torch_eager_fp32=22862.630675, rel_fro_vs_torch_eager_fp32=0.055222 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 3.2 | 3.2..3.2 | 1 | 4.731 | - | - | 3.2 | - (stored kernel) | 4.731 (MIXED ours whole / arm kernel) | - | 3723.1 | 501.4 | max_rel_diff_vs_torch_eager_fp32=22880.029448, rel_fro_vs_torch_eager_fp32=0.055199 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 67.2 | 67.2..67.2 | 1 | - | - | 67.2 | - | - (stored whole) | - | - | - | - | accuracy=0.853620, logloss=3.628565 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 209.1 | 209.1..209.1 | 1 | 0.321 | - | 209.1 | 209.1 | 0.00 (cpu-arm) | 0.321 (whole/whole) | - | 3958.7 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 27.9 | 27.9..27.9 | 1 | - | - | 27.9 | - | - (stored whole) | - | - | - | - | accuracy=0.723160, logloss=0.590725 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38.6 | 38.6..38.6 | 1 | 0.724 | - | 38.6 | 38.6 | 0.00 (cpu-arm) | 0.724 (whole/whole) | - | 450.2 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0879)`, ran on AMD Instinct MI325X (amd, DO) job a0879

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 17.7 | 17.7..17.7 | 1 | - | - | 17.7 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559529 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0879 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 271.3 | 271.3..271.3 | 1 | 0.065 | - | 271.3 | 271.3 | 0.00 (cpu-arm) | 0.065 (whole/whole) | - | 4455.8 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nadam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 7.0 ms) = 0.575. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 4.0 | 4.0..4.0 | 1 | - | - | 4.0 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 7.0 | 7.0..7.0 | 1 | 0.575 | - | - | 7.0 | - (stored kernel) | 0.575 (MIXED ours whole / arm kernel) | - | 2835.0 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 333.5 | 333.5..333.5 | 1 | 0.012 | - | - | 333.5 | - (stored kernel) | 0.012 (MIXED ours whole / arm kernel) | - | 2927.7 | 896.0 | rel_fro_vs_torch_eager_fp32=1.464e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 22.5 | 22.5..22.5 | 1 | - | - | 22.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x119 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 13.3 | 13.3..13.3 | 1 | 1.701 | - | 13.3 | 13.3 | 0.00 (cpu-arm) | 1.701 (whole/whole) | - | 539.3 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 19.1 | 19.1..19.1 | 1 | - | - | 19.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x508 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 8.0 | 8.0..8.0 | 1 | 2.374 | - | 8.0 | 8.0 | 0.00 (cpu-arm) | 2.374 (whole/whole) | - | 1407.9 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on AMD Instinct MI325X (amd, DO) job a1069

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 626.5 | 626.5..626.5 | 1 | - | - | 626.5 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.438855 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 303.9 | 303.9..303.9 | 1 | 2.061 | - | 303.9 | 303.9 | 0.00 (cpu-arm) | 2.061 (whole/whole) | - | 283.6 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on AMD Instinct MI325X (amd, DO) job a1069

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 1116.7 | 1116.7..1116.7 | 1 | - | - | 1116.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.150860 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 770.4 | 770.4..770.4 | 1 | 1.450 | - | 770.4 | 770.4 | 0.00 (cpu-arm) | 1.450 (whole/whole) | - | 283.5 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 22.1 | 22.1..22.1 | 1 | - | - | 22.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 13.6 | 13.6..13.6 | 1 | 1.631 | - | 13.6 | 13.6 | 0.00 (cpu-arm) | 1.631 (whole/whole) | - | 267.7 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 18.5 | 18.5..18.5 | 1 | - | - | 18.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 7.4 | 7.4..7.4 | 1 | 2.484 | - | 7.4 | 7.4 | 0.00 (cpu-arm) | 2.484 (whole/whole) | - | 246.0 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 2.4 | 2.4..2.4 | 1 | - | - | 2.4 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 115.0 | 115.0..115.0 | 1 | 0.021 | - | 115.0 | 115.0 | 0.00 (cpu-arm) | 0.021 (whole/whole) | - | 213.2 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on AMD Instinct MI325X (amd, DO) job a0870

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 2.3 | 2.3..2.3 | 1 | - | - | 2.3 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | networkx 3.6.1 | opponent | 88.5 | 88.5..88.5 | 1 | 0.026 | - | 88.5 | 88.5 | 0.00 (cpu-arm) | 0.026 (whole/whole) | - | 178.4 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on AMD Instinct MI325X (amd, DO) job a0904

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 1979.0 | 1979.0..1979.0 | 1 | - | - | 1979.0 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 22453.9 | 22453.9..22453.9 | 1 | 0.088 | - | 22453.9 | 22453.9 | 0.00 (cpu-arm) | 0.088 (whole/whole) | - | 1842.9 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on AMD Instinct MI325X (amd, DO) job a0904

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 31.8 | 31.8..31.8 | 1 | - | - | 31.8 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 104.1 | 104.1..104.1 | 1 | 0.305 | - | 104.1 | 104.1 | 0.00 (cpu-arm) | 0.305 (whole/whole) | - | 478.9 | - | rel_error_vs_exact=1.218e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on AMD Instinct MI325X (amd, DO) job a0868

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 724.8 | 724.8..724.8 | 1 | - | - | 724.8 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.485e-07 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1852.3 | 1852.3..1852.3 | 1 | 0.391 | - | - | 1852.3 | - (stored kernel) | 0.391 (MIXED ours whole / arm kernel) | - | 3625.8 | 2520.6 | relative_gram_difference=0.0001698 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 4646.6 | 4646.6..4646.6 | 1 | 0.156 | - | 4646.6 | 4646.6 | 0.00 (cpu-arm) | 0.156 (whole/whole) | - | 8532.1 | - | relative_gram_difference=2.462e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on AMD Instinct MI325X (amd, DO) job a0868

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 16.5 | 16.5..16.5 | 1 | - | - | 16.5 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.714e-07 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 24.6 | 24.6..24.6 | 1 | 0.669 | - | - | 24.6 | - (stored kernel) | 0.669 (MIXED ours whole / arm kernel) | - | 2690.6 | 126.0 | relative_gram_difference=6.794e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 177.3 | 177.3..177.3 | 1 | 0.093 | - | 177.3 | 177.3 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 478.4 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0535)`, ran on AMD Instinct MI325X (amd, DO) job a0535

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 862.1 | 862.1..862.1 | 1 | - | - | 862.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.039998, rmse=0.851877 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0535 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 878876.6 | 878876.6..878876.6 | 1 | 0.0009809 | - | 878876.6 | 878876.6 | 0.00 (cpu-arm) | 0.0009809 (whole/whole) | - | 8006.1 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0535)`, ran on AMD Instinct MI325X (amd, DO) job a0535

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 499.3 | 499.3..499.3 | 1 | - | - | 499.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.899596, rmse=5.046749 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0535 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 225040.2 | 225040.2..225040.2 | 1 | 0.002 | - | 225040.2 | 225040.2 | 0.00 (cpu-arm) | 0.002 (whole/whole) | - | 928.9 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1554)`, ran on AMD Instinct MI325X (amd, DO) job a1554

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 125.6 | 125.6..125.6 | 1 | - | - | 125.6 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.0002359 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1554 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 352.0 | 352.0..352.0 | 1 | 0.357 | - | - | 352.0 | - (stored kernel) | 0.357 (MIXED ours whole / arm kernel) | - | 3836.8 | 1017.6 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 812.6 | 812.6..812.6 | 1 | 0.155 | - | 812.6 | 812.6 | 0.00 (cpu-arm) | 0.155 (whole/whole) | - | 1710.8 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1554)`, ran on AMD Instinct MI325X (amd, DO) job a1554

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 47.4 | 47.4..47.4 | 1 | - | - | 47.4 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.027197 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1554 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 215.0 | 215.0..215.0 | 1 | 0.220 | - | - | 215.0 | - (stored kernel) | 0.220 (MIXED ours whole / arm kernel) | - | 3037.1 | 228.0 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 375.4 | 375.4..375.4 | 1 | 0.126 | - | 375.4 | 375.4 | 0.00 (cpu-arm) | 0.126 (whole/whole) | - | 476.9 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.1 ms) = 7.301; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.2 ms) = 6.811. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 15.0 | 15.0..15.0 | 1 | - | - | 15.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1200/work-resnet-block-synthetic-def/resnet-block-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.2 | 2.2..2.2 | 1 | 6.811 | - | - | 2.2 | - (stored kernel) | 6.811 (MIXED ours whole / arm kernel) | - | 2892.9 | 543.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.2 | 2.2..2.2 | 1 | 6.763 | - | - | 2.2 | - (stored kernel) | 6.763 (MIXED ours whole / arm kernel) | - | 3084.9 | 505.1 | max_rel_diff_vs_torch_eager_fp32=0.834465, rel_fro_vs_torch_eager_fp32=1.645e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.1 | 2.1..2.1 | 1 | 7.301 | - | - | 2.1 | - (stored kernel) | 7.301 (MIXED ours whole / arm kernel) | - | 3181.3 | 468.0 | max_rel_diff_vs_torch_eager_fp32=28679.370880, rel_fro_vs_torch_eager_fp32=0.004232 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2.4 | 2.4..2.4 | 1 | 6.272 | - | - | 2.4 | - (stored kernel) | 6.272 (MIXED ours whole / arm kernel) | - | 3029.2 | 430.5 | max_rel_diff_vs_torch_eager_fp32=18454.670906, rel_fro_vs_torch_eager_fp32=0.003485 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 62958.7 | 62958.7..62958.7 | 1 | - | - | 62958.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328684, rmse=0.684422 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 78354.8 | 78354.8..78354.8 | 1 | 0.804 | - | 78354.8 | 78354.8 | 0.00 (cpu-arm) | 0.804 (whole/whole) | - | 8705.1 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 288.9 | 288.9..288.9 | 1 | - | - | 288.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 994.6 | 994.6..994.6 | 1 | 0.290 | - | 994.6 | 994.6 | 0.00 (cpu-arm) | 0.290 (whole/whole) | - | 380.7 | - | finite=True, r2=0.908983, rmse=4.805057 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.1 ms) = 0.659. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 5.1 | 5.1..5.1 | 1 | 0.659 | - | - | 5.1 | - (stored kernel) | 0.659 (MIXED ours whole / arm kernel) | - | 2833.1 | 896.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 281.9 | 281.9..281.9 | 1 | 0.012 | - | - | 281.9 | - (stored kernel) | 0.012 (MIXED ours whole / arm kernel) | - | 2924.1 | 832.0 | rel_fro_vs_torch_eager_fp32=6.375e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 950.8 ms) = 0.322; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 879.7 ms) = 0.348. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 305.9 | 305.9..305.9 | 1 | - | - | 305.9 | - | - (stored whole) | - | - | - | - | accuracy=0.967828, logloss=0.079029 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 889.2 | 889.2..889.2 | 1 | 0.344 | - | - | 889.2 | - (stored kernel) | 0.344 (MIXED ours whole / arm kernel) | - | 3215.3 | 110.3 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 879.7 | 879.7..879.7 | 1 | 0.348 | - | - | 879.7 | - (stored kernel) | 0.348 (MIXED ours whole / arm kernel) | - | 3255.8 | 110.3 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 950.8 | 950.8..950.8 | 1 | 0.322 | - | - | 950.8 | - (stored kernel) | 0.322 (MIXED ours whole / arm kernel) | - | 4965.1 | 98.7 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 995.6 | 995.6..995.6 | 1 | 0.307 | - | - | 995.6 | - (stored kernel) | 0.307 (MIXED ours whole / arm kernel) | - | 5005.9 | 98.7 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 975.1 ms) = 0.314; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 743.5 ms) = 0.412. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 306.6 | 306.6..306.6 | 1 | - | - | 306.6 | - | - (stored whole) | - | - | - | - | accuracy=0.862684, logloss=0.313008 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 743.5 | 743.5..743.5 | 1 | 0.412 | - | - | 743.5 | - (stored kernel) | 0.412 (MIXED ours whole / arm kernel) | - | 3478.2 | 110.3 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 885.8 | 885.8..885.8 | 1 | 0.346 | - | - | 885.8 | - (stored kernel) | 0.346 (MIXED ours whole / arm kernel) | - | 3256.1 | 110.3 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 975.1 | 975.1..975.1 | 1 | 0.314 | - | - | 975.1 | - (stored kernel) | 0.314 (MIXED ours whole / arm kernel) | - | 5234.9 | 98.7 | accuracy=0.867947 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1002.0 | 1002.0..1002.0 | 1 | 0.306 | - | - | 1002.0 | - (stored kernel) | 0.306 (MIXED ours whole / arm kernel) | - | 5006.2 | 98.7 | accuracy=0.867947 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 898.5 ms) = 0.334; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 891.9 ms) = 0.337. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 300.1 | 300.1..300.1 | 1 | - | - | 300.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.978602, rmse=0.169476 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 891.9 | 891.9..891.9 | 1 | 0.337 | - | - | 891.9 | - (stored kernel) | 0.337 (MIXED ours whole / arm kernel) | - | 3040.0 | 109.9 | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 901.6 | 901.6..901.6 | 1 | 0.333 | - | - | 901.6 | - (stored kernel) | 0.333 (MIXED ours whole / arm kernel) | - | 3082.4 | 109.9 | finite=True, r2=0.977348, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 898.5 | 898.5..898.5 | 1 | 0.334 | - | - | 898.5 | - (stored kernel) | 0.334 (MIXED ours whole / arm kernel) | - | 3663.2 | 98.4 | finite=True, r2=0.977371, rmse=0.174283 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 953.5 | 953.5..953.5 | 1 | 0.315 | - | - | 953.5 | - (stored kernel) | 0.315 (MIXED ours whole / arm kernel) | - | 3705.5 | 98.4 | finite=True, r2=0.977371, rmse=0.174283 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1010.8 ms) = 0.297; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 762.7 ms) = 0.394. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 300.7 | 300.7..300.7 | 1 | - | - | 300.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.743904, rmse=0.548825 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 863.9 | 863.9..863.9 | 1 | 0.348 | - | - | 863.9 | - (stored kernel) | 0.348 (MIXED ours whole / arm kernel) | - | 3041.2 | 109.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 762.7 | 762.7..762.7 | 1 | 0.394 | - | - | 762.7 | - (stored kernel) | 0.394 (MIXED ours whole / arm kernel) | - | 3082.6 | 109.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1010.8 | 1010.8..1010.8 | 1 | 0.297 | - | - | 1010.8 | - (stored kernel) | 0.297 (MIXED ours whole / arm kernel) | - | 3664.6 | 98.4 | finite=True, r2=0.739017, rmse=0.554037 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 1033.9 | 1033.9..1033.9 | 1 | 0.291 | - | - | 1033.9 | - (stored kernel) | 0.291 (MIXED ours whole / arm kernel) | - | 3705.5 | 98.4 | finite=True, r2=0.739017, rmse=0.554037 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on AMD Instinct MI325X (amd, DO) job a1065

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 3961.9 | 3961.9..3961.9 | 1 | - | - | 3961.9 | - | - (stored whole) | - | - | - | - | accuracy=0.920330 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 28098.7 | 28098.7..28098.7 | 1 | 0.141 | - | 28098.7 | 28098.7 | 0.00 (cpu-arm) | 0.141 (whole/whole) | - | 1143.5 | - | accuracy=0.910200 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on AMD Instinct MI325X (amd, DO) job a1065

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 2320.4 | 2320.4..2320.4 | 1 | - | - | 2320.4 | - | - (stored whole) | - | - | - | - | accuracy=0.755330 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 8379.1 | 8379.1..8379.1 | 1 | 0.277 | - | 8379.1 | 8379.1 | 0.00 (cpu-arm) | 0.277 (whole/whole) | - | 266.7 | - | accuracy=0.752520 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 23.4 | 23.4..23.4 | 1 | - | - | 23.4 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 480.0 | 480.0..480.0 | 1 | 0.049 | - | 480.0 | 480.0 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 3271.2 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 60.6 | 60.6..60.6 | 1 | 0.056 | - | 60.6 | 60.6 | 0.00 (cpu-arm) | 0.056 (whole/whole) | - | 365.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on AMD Instinct MI325X (amd, DO) job a0868

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 741.9 | 741.9..741.9 | 1 | - | - | 741.9 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=20.498972, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 2159.2 | 2159.2..2159.2 | 1 | 0.344 | - | - | 2159.2 | - (stored kernel) | 0.344 (MIXED ours whole / arm kernel) | - | 3796.1 | 3361.7 | max_rel_singular_value_error=2.707e+08, relative_reconstruction_error_100k_rows=0.026534 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 3741.4 | 3741.4..3741.4 | 1 | 0.198 | - | 3741.4 | 3741.4 | 0.00 (cpu-arm) | 0.198 (whole/whole) | - | 8708.9 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on AMD Instinct MI325X (amd, DO) job a0868

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 19.1 | 19.1..19.1 | 1 | - | - | 19.1 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.6.0+rocm6.4.1.git1ded221d | opponent | 271.6 | 271.6..271.6 | 1 | 0.070 | - | - | 271.6 | - (stored kernel) | 0.070 (MIXED ours whole / arm kernel) | - | 2750.5 | 168.0 | max_rel_singular_value_error=4.401e-05, relative_reconstruction_error_100k_rows=0.003043 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 130.5 | 130.5..130.5 | 1 | 0.146 | - | 130.5 | 130.5 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 485.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0495)`, ran on AMD Instinct MI325X (amd, DO) job a0495

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 1091.7 | 1091.7..1091.7 | 1 | - | - | 1091.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0495 2026-10-08; identity vs nvidia-l40s: MATCH) |
| gpytorch-gpu | gpytorch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | gpytorch 1.15.2 | opponent | 48.7 | 48.7..48.7 | 1 | 22.398 | - | - | 48.7 | - (stored kernel) | 22.398 (MIXED ours whole / arm kernel) | - | 4591.1 | 1611.8 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | gpytorch 1.15.2 | opponent | 672.0 | 672.0..672.0 | 1 | 1.625 | - | 672.0 | 672.0 | 0.00 (cpu-arm) | 1.625 (whole/whole) | - | 3457.1 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 89.7 | 89.7..89.7 | 1 | - | - | 89.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 202.8 | 202.8..202.8 | 1 | 0.442 | - | 202.8 | 202.8 | 0.00 (cpu-arm) | 0.442 (whole/whole) | - | 348.0 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on AMD Instinct MI325X (amd, DO) job a1068

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 82.9 | 82.9..82.9 | 1 | - | - | 82.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 116.5 | 116.5..116.5 | 1 | 0.712 | - | 116.5 | 116.5 | 0.00 (cpu-arm) | 0.712 (whole/whole) | - | 307.5 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on AMD Instinct MI325X (amd, DO) job a1069

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 189.2 | 189.2..189.2 | 1 | - | - | 189.2 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436610 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 104.8 | 104.8..104.8 | 1 | 1.806 | - | 104.8 | 104.8 | 0.00 (cpu-arm) | 1.806 (whole/whole) | - | 283.6 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 38.6 | 38.6..38.6 | 1 | 4.903 | - | 38.6 | 38.6 | 0.00 (cpu-arm) | 4.903 (whole/whole) | - | 57.6 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on AMD Instinct MI325X (amd, DO) job a1069

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8ed94710a) | identical | 596.1 | 596.1..596.1 | 1 | - | - | 596.1 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.020604 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 548.4 | 548.4..548.4 | 1 | 1.087 | - | 548.4 | 548.4 | 0.00 (cpu-arm) | 1.087 (whole/whole) | - | 283.4 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 52.3 | 52.3..52.3 | 1 | 11.399 | - | 52.3 | 52.3 | 0.00 (cpu-arm) | 11.399 (whole/whole) | - | 58.1 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / istella (rows full, shape X 100000x220; Xq 10000x220; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0877)`, ran on AMD Instinct MI325X (amd, DO) job a0877

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | max_additivity_error=1.175e-06 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0877 2026-10-08; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 415.9 | 415.9..415.9 | 1 | 0.020 | - | 415.9 | 415.9 | 0.00 (cpu-arm) | 0.020 (whole/whole) | - | 1468.4 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 344.0 | 344.0..344.0 | 1 | 0.025 | - | 344.0 | 344.0 | 0.00 (cpu-arm) | 0.025 (whole/whole) | - | 1345.6 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 275.4 | 275.4..275.4 | 1 | 0.031 | - | 275.4 | 275.4 | 0.00 (cpu-arm) | 0.031 (whole/whole) | - | 1532.8 | - | max_additivity_error=4.441e-15 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / taxi (rows full, shape X 100000x11; Xq 10000x11; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0877)`, ran on AMD Instinct MI325X (amd, DO) job a0877

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 2.8 | 2.8..2.8 | 1 | - | - | 2.8 | - | - (stored whole) | - | - | - | - | max_additivity_error=3.858e-05 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0877 2026-10-08; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 161.7 | 161.7..161.7 | 1 | 0.018 | - | 161.7 | 161.7 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 399.3 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 156.1 | 156.1..156.1 | 1 | 0.018 | - | 156.1 | 156.1 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 286.1 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 117.1 | 117.1..117.1 | 1 | 0.024 | - | 117.1 | 117.1 | 0.00 (cpu-arm) | 0.024 (whole/whole) | - | 269.5 | - | max_additivity_error=5.684e-13 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1192)`, ran on AMD Instinct MI325X (amd, DO) job a1192

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 1215.0 | 1215.0..1215.0 | 1 | - | - | 1215.0 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.992124 | - | main board, one scored run | - | ok (main@ca8ea1f8d amd/a1192 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 17896.2 | 17896.2..17896.2 | 1 | 0.068 | - | 17896.2 | 17896.2 | 0.00 (cpu-arm) | 0.068 (whole/whole) | - | 370.6 | - | trustworthiness_k15=0.992170 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1192)`, ran on AMD Instinct MI325X (amd, DO) job a1192

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 1048.3 | 1048.3..1048.3 | 1 | - | - | 1048.3 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.998921 | - | main board, one scored run | - | ok (main@ca8ea1f8d amd/a1192 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 15382.8 | 15382.8..15382.8 | 1 | 0.068 | - | 15382.8 | 15382.8 | 0.00 (cpu-arm) | 0.068 (whole/whole) | - | 379.6 | - | trustworthiness_k15=0.998823 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

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
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), ragged `lengths` and a prefill followed by decode are public and not raced; the zero-state forward (*-forward) and the zero-state token-by-token decode (*-decode) are.
- Neural: The byte LM has no incremental decode on any route: LanguageModelTrainer (GPU) and LanguageModelInference (CPU) expose full-sequence logits only (lm-forward on the GPU), no KV-cache state or step, so there is no lm-decode row.
- Neural: The *-infer and lm-host-train-step rows are the CPU host binding and are never raced; their GPU twins are the *-forward, *-decode and mlp-predict rows.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on ROCm accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: mamba1-decode, mamba2-decode, mamba3-decode and samba-decode race ours alone: the repo's torch Mamba references are full-sequence scans with no carried-state decode step (a torch decode twin is not written yet)
- Neural, not planned on this vendor: torch-compile-* on transformer-decode: the twin is a per-token loop over a growing KV cache; transformer-decode races the eager arms only
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on ROCm has no int8 matmul, so ours races alone

