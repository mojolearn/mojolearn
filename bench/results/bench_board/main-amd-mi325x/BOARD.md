# mojolearn benchmark board

Generated 2026-10-11T21:57:51Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD amd-mi325x, version label main@7924718fc. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 181 races; oldest cell main@89485bc96 (2026-10-08T13:27:06Z), newest cell main@7924718fc (2026-10-11T21:48:42Z). Boxes: AMD Instinct MI325X (amd, DO).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 127 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: one scored run per cell (lq RACE ALGOS lines, lq CMD bench_board summaries); the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 118, n/a 63).

> Opponents: copied from the stored opponent boards (release-board-resume-r2), never re-run here; `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 2 races.

> Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax. A cell of ours whose output hash equals a copied opponent's shows quality identical_to=<arm> (the same bits) where the own-host reference gave none.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 118, n/a 63.

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
| mojolearn | main@7924718fc (wheel none (unreleased; built from source at each cell's commit), sha256 -) |
| script commit | 7924718fc95688b88982a9ea06fa5dcd2c3ed93e |
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

Races: 181 planned, 181 done, 0 failed, 0 unsupported, 0 pending. Cells: 533 (HOST-MEMORY 1, REFUSED 9, ok 523).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adamax | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.17e-09 |
| algos | ard | istella | r2 (higher is better) | - | -0.124856 | sklearn-cpu 0.327436 |
| algos | ard | istella | rmse (lower is better) | - | 0.885949 | sklearn-cpu 0.685058 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002793; torch-eager-bf16 0.000000; torch-compile-bf16 0.002793 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.248e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.248e-08 |
| algos | batchnorm2d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001471; torch-eager-bf16 0.000000; torch-compile-bf16 0.001471 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.868e-08; torch-eager-bf16 0.000000; torch-compile-bf16 6.868e-08 |
| algos | bayesian-ridge | istella | r2 (higher is better) | - | 0.317909 | sklearn-cpu -8.145e+07 |
| algos | bayesian-ridge | istella | rmse (lower is better) | - | 0.689893 | sklearn-cpu 7538.692925 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | - | 0.794050 | sklearn-cpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | - | 5.350625 | sklearn-cpu 4.278741 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | 0.755560 | sklearn-cpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | 0.557803 | sklearn-cpu 0.557802 |
| algos | categorical-nb | istella | accuracy (higher is better) | - | 0.838850 | sklearn-cpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | - | 0.412625 | sklearn-cpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | 0.765850 | sklearn-cpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | 0.538866 | sklearn-cpu 0.538866 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | numpy-cpu 3.928e-08; torch-gpu 2.519e-07 |
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
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 3418.486565; torch-compile-bf16 3418.486565 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.003382; torch-compile-bf16 0.003382 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | - | 13.931153 | statsmodels-cpu 26.588738; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | - | 96.690449 | statsmodels-cpu 196.955273; statsforecast-cpu 96.685568 |
| algos | dynamic-optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.436278 | statsforecast-cpu 1.436045 |
| algos | dynamic-optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.086628 | statsforecast-cpu 49.314797 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 3.692e-07 | numpy-cpu 3.49e-08; torch-gpu 1.044e-06 |
| algos | eigh | synthetic | relative_residual | - | 6.672e-07 | numpy-cpu 2.824e-08; torch-gpu 9.614e-07 |
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
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.007740; torch-eager-bf16 2866.979199; torch-compile-bf16 2866.972319 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.044e-08; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002794; torch-eager-bf16 1509.504160; torch-compile-bf16 1509.503229 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 7.648e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.007299; torch-eager-bf16 0.000000; torch-compile-bf16 0.007299 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.01e-07; torch-eager-bf16 0.000000; torch-compile-bf16 1.01e-07 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.108108; torch-eager-bf16 3907.203674; torch-compile-bf16 3678.128123 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 8.065e-08; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.074506; torch-eager-bf16 5040.230769; torch-compile-bf16 4607.677460 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 4.566e-08; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | 0.971625 | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-bf16 0.971680; torch-compile-bf16 0.971680 |
| algos | gru-clf | synthetic | logloss (lower is better) | - | 0.070054 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | 0.865723 | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-bf16 0.865777; torch-compile-bf16 0.865777 |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | - | 0.303537 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | - | 0.982337 | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-bf16 0.981981; torch-compile-bf16 0.981981 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | 0.153975 | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-bf16 0.155521; torch-compile-bf16 0.155521 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | 0.747875 | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-bf16 0.748253; torch-compile-bf16 0.748253 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | 0.544554 | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-bf16 0.544145; torch-compile-bf16 0.544145 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | 0.802100 | faiss-cpu 0.801550 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.981250 | faiss-cpu 0.979450 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | - | 0.994575 | faiss-cpu 0.993125 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | 8.307e-08 | shap-cpu 8.63e-15 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | 1.056e-07 | shap-cpu 8.149e-15 |
| algos | knn-imputer | istella | masked_rmse | - | 323953.237332 | sklearn-cpu 986208.700423 |
| algos | knn-imputer | taxi | masked_rmse | - | 6.151696 | sklearn-cpu 5.263919 |
| algos | lars | istella | r2 (higher is better) | - | 0.328699 | sklearn-cpu -1.051e+22 |
| algos | lars | istella | rmse (lower is better) | - | 0.684414 | sklearn-cpu 8.562e+10 |
| algos | lars | taxi | r2 (higher is better) | - | 0.908983 | sklearn-cpu 0.908983 |
| algos | lars | taxi | rmse (lower is better) | - | 4.805051 | sklearn-cpu 4.805057 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.325506 | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.686040 | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909038 | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803593 | sklearn-cpu 4.803593 |
| algos | lasso-lars | istella | r2 (higher is better) | - | 0.310834 | sklearn-cpu 0.310591 |
| algos | lasso-lars | istella | rmse (lower is better) | - | 0.693461 | sklearn-cpu 0.693584 |
| algos | lasso-lars | taxi | r2 (higher is better) | - | 0.908998 | sklearn-cpu 0.908998 |
| algos | lasso-lars | taxi | rmse (lower is better) | - | 4.804663 | sklearn-cpu 4.804669 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.007530; torch-eager-bf16 0.000000; torch-compile-bf16 0.007530 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.326e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.326e-08 |
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
| algos | lstm-reg | synthetic | rmse (lower is better) | - | 0.166295 | torch-eager-fp32 0.159642; torch-compile-fp32 0.159642; torch-eager-bf16 0.159679; torch-compile-bf16 0.159679 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | 0.754306 | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-bf16 0.751591; torch-compile-bf16 0.751591 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | 0.537563 | torch-eager-fp32 0.540429; torch-compile-fp32 0.540429; torch-eager-bf16 0.540526; torch-compile-bf16 0.540526 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | numpy-cpu 0.876106; torch-gpu nan |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | numpy-cpu 0.756366; torch-gpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | scipy-cpu 3.439e-07; torch-gpu 4.003e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | numpy-cpu 3.259e-08; torch-gpu 4.003e-07 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | minibatch-kmeans | istella | n_clusters | - | 8 | sklearn-cpu 8 |
| algos | minibatch-kmeans | istella | silhouette (higher is better) | - | 0.116696 | sklearn-cpu 0.111849 |
| algos | minibatch-kmeans | istella | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 0.622446 |
| algos | minibatch-kmeans | taxi | n_clusters | - | 8 | sklearn-cpu 8 |
| algos | minibatch-kmeans | taxi | silhouette (higher is better) | - | 0.138060 | sklearn-cpu 0.165473 |
| algos | minibatch-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 0.525151 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.017192; torch-eager-bf16 22778.779951; torch-compile-bf16 22796.112469 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.088e-07; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | multinomial-nb | istella | accuracy (higher is better) | - | 0.853620 | sklearn-cpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | - | 3.628565 | sklearn-cpu 3.087499 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | - | 0.723160 | sklearn-cpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | - | 0.590725 | sklearn-cpu 0.590725 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | sklearn-cpu 0.557319 |
| algos | nadam | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.464e-07 |
| algos | nearest-centroid | istella | accuracy (higher is better) | - | 0.852610 | sklearn-cpu 0.852610 |
| algos | nearest-centroid | istella | logloss (lower is better) | - | 4.299485 | sklearn-cpu 4.117692 |
| algos | nearest-centroid | taxi | accuracy (higher is better) | - | 0.666700 | sklearn-cpu 0.666750 |
| algos | nearest-centroid | taxi | logloss (lower is better) | - | 0.781905 | sklearn-cpu 0.781690 |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.438855 | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.150860 | statsforecast-cpu 49.356608 |
| algos | pagerank | istella | sum | - | 1.000000 | networkx-cpu 1.000000 |
| algos | pagerank | taxi | sum | - | 1.000000 | networkx-cpu 1.000000 |
| algos | permutation-shap | istella | rel_error_vs_exact | - | 6.68e-08 | shap-cpu 3.692e-10 |
| algos | permutation-shap | taxi | rel_error_vs_exact | - | 1.054e-07 | shap-cpu 1.218e-15 |
| algos | qr | istella | relative_gram_difference | - | 1.485e-07 | numpy-cpu 2.462e-08; torch-gpu 0.0001695 |
| algos | qr | taxi | relative_gram_difference | - | 1.714e-07 | numpy-cpu 3.024e-08; torch-gpu 9.54e-07 |
| algos | quantile | istella | r2 (higher is better) | - | -0.039998 | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | - | 0.851877 | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | - | 0.899596 | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | - | 5.046749 | sklearn-cpu 5.044706 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | 0.0002359 | sklearn-cpu 0.0002359; torch-gpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | 0.027197 | sklearn-cpu 0.027197; torch-gpu 0.027197 |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.357628; torch-eager-bf16 24577.140808; torch-compile-bf16 18499.135971 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.809e-07; torch-eager-bf16 0.004174; torch-compile-bf16 0.003425 |
| algos | ridge-cv | istella | r2 (higher is better) | - | 0.328684 | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | 0.684422 | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | 0.908981 | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | 4.805109 | sklearn-cpu 4.805057 |
| algos | rmsprop | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | 0.967828 | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-bf16 0.953396; torch-compile-bf16 0.953396 |
| algos | rnn-clf | synthetic | logloss (lower is better) | - | 0.079029 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | 0.862684 | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-bf16 0.868056; torch-compile-bf16 0.868056 |
| algos | rnn-clf | taxi-hourly | logloss (lower is better) | - | 0.313008 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | r2 (higher is better) | - | 0.978602 | torch-eager-fp32 0.977348; torch-compile-fp32 0.977348; torch-eager-bf16 0.977383; torch-compile-bf16 0.977383 |
| algos | rnn-reg | synthetic | rmse (lower is better) | - | 0.169476 | torch-eager-fp32 0.174374; torch-compile-fp32 0.174374; torch-eager-bf16 0.174238; torch-compile-bf16 0.174238 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | 0.743904 | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-bf16 0.738886; torch-compile-bf16 0.738886 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | 0.548825 | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-bf16 0.554176; torch-compile-bf16 0.554176 |
| algos | sgd-clf | istella | accuracy (higher is better) | - | 0.920330 | sklearn-cpu 0.910200 |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | 0.755330 | sklearn-cpu 0.752520 |
| algos | svd | istella | max_rel_singular_value_error | - | 20.498972 | numpy-cpu 1.000000; torch-gpu 2.707e+08 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | 3.379e-05 | numpy-cpu 4.1e-08; torch-gpu 0.026534 |
| algos | svd | taxi | max_rel_singular_value_error | - | 7.036e-07 | numpy-cpu 4.308e-08; torch-gpu 4.409e-05 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | 1.18e-06 | numpy-cpu 4.314e-08; torch-gpu 0.003043 |
| algos | svgp | istella | r2 (higher is better) | - | -0.106016 | gpytorch-gpu -0.106040; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | 0.878373 | gpytorch-gpu 0.878383; gpytorch-cpu 0.878383 |
| algos | svgp | taxi | r2 (higher is better) | - | -0.194982 | gpytorch-gpu -0.209526; gpytorch-cpu -0.209532 |
| algos | svgp | taxi | rmse (lower is better) | - | 17.723482 | gpytorch-gpu 17.831014; gpytorch-cpu 17.831057 |
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
| classical | kde | istella | mean_log_likelihood (higher is better) | - | -222.270583 | sklearn-cpu -226.977407 |
| classical | kde | istella | rows_without_density | - | 0 | sklearn-cpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | - | -14.826436 | sklearn-cpu -14.826437 |
| classical | kde | taxi | rows_without_density | - | 0 | sklearn-cpu 0 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | torch-gpu 5.991e+17; sklearn-cpu 6.049e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | torch-gpu 0.990156; sklearn-cpu 0.999759 |
| classical | kmeans | istella | n_iter | - | 33 | torch-gpu 57; sklearn-cpu 24 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | torch-gpu 3.129e+08; sklearn-cpu 3.093e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | torch-gpu 1.011535; sklearn-cpu 0.999937 |
| classical | kmeans | taxi | n_iter | - | 91 | torch-gpu 80; sklearn-cpu 58 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | torch-gpu 0.979840; sklearn-cpu 1.000000 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | torch-gpu 0; sklearn-cpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | torch-gpu 0.999738; sklearn-cpu 1.000000 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | torch-gpu 0; sklearn-cpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | torch-gpu nan; torch-gpu-eigh 0.151604; sklearn-cpu 0.001881 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | torch-gpu nan; torch-gpu-eigh 0.768590; sklearn-cpu 0.833655 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908837 | torch-gpu 0.908837; torch-gpu-eigh 0.908827; sklearn-cpu 0.724850 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696477 | torch-gpu 4.696472; torch-gpu-eigh 4.696715; sklearn-cpu 8.159187 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | torch-gpu 1.000000; sklearn-cpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999997 | torch-gpu 0.999996; sklearn-cpu 0.999995 |
| classical | svc | istella | accuracy (higher is better) | - | 0.922200 | sklearn-cpu 0.922200 |
| classical | svc | istella | n_support | - | 2400 | sklearn-cpu 2400 |
| classical | svc | taxi | accuracy (higher is better) | - | 0.767500 | sklearn-cpu 0.767500 |
| classical | svc | taxi | n_support | - | 5527 | sklearn-cpu 5675 |
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
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 0.001038; torch-compile-fp32 0.001038; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 2.866e-06; torch-compile-fp32 2.866e-06; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | torch-eager-fp32 9.018733; torch-compile-fp32 9.018733; torch-eager-bf16 9.018647; torch-compile-bf16 9.018653 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.222e-06; torch-compile-fp32 1.192e-06; torch-eager-bf16 0.006313; torch-compile-bf16 0.006870 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.677e-06; torch-compile-fp32 1.637e-06; torch-eager-bf16 0.008666; torch-compile-bf16 0.009432 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | torch-eager-fp32 9.018732; torch-compile-fp32 9.018734; torch-eager-bf16 9.018677; torch-compile-bf16 9.018653 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.418446 | torch-eager-fp32 8.422411; torch-compile-fp32 8.422412; torch-eager-bf16 8.420410; torch-compile-bf16 8.419378 |
| neural | lm-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | lm-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 9.537e-07; torch-compile-fp32 9.537e-07; torch-eager-bf16 5.627e-05; torch-compile-bf16 8.011e-05 |
| neural | lm-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 9.537e-07; torch-eager-bf16 0.002001; torch-compile-bf16 0.003033 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 2.384e-07; torch-eager-bf16 5.436e-05 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.189e-07; torch-eager-bf16 2.711e-05 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.073e-06; torch-compile-fp32 1.073e-06; torch-eager-bf16 0.007144; torch-compile-bf16 0.007144 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 3.648e-07; torch-compile-fp32 3.648e-07; torch-eager-bf16 0.002429; torch-compile-bf16 0.002429 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | torch-eager-fp32 1.160401; torch-compile-fp32 1.160401; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | torch-eager-fp32 1.123361; torch-compile-fp32 1.123361; torch-eager-bf16 1.123461; torch-compile-bf16 1.123461 |
| neural | mlp-train-step | gaussian | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 9.656e-05; torch-compile-bf16 9.656e-05 |
| neural | mlp-train-step | gaussian | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 0.0001005; torch-compile-bf16 0.0001006 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | torch-eager-fp32 5.635910; torch-compile-fp32 5.635910; torch-eager-bf16 5.635950; torch-compile-bf16 5.635933 |
| neural | samba-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 2.146e-06; torch-compile-fp32 2.168e-06; torch-eager-bf16 0.017180; torch-compile-bf16 0.016678 |
| neural | samba-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.206e-06; torch-compile-fp32 1.219e-06; torch-eager-bf16 0.009659; torch-compile-bf16 0.009377 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | torch-eager-fp32 5.635910; torch-compile-fp32 5.635909; torch-eager-bf16 5.635949; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | torch-eager-fp32 4.833934; torch-compile-fp32 4.833934; torch-eager-bf16 4.834015; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 9.537e-07; torch-eager-bf16 3.91e-05; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 8.106e-05; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.001819; torch-compile-bf16 0.001819 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.02e-07; torch-compile-fp32 1.02e-07; torch-eager-bf16 0.0003893; torch-compile-bf16 0.0003893 |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | 0.631297 | catboost-cpu 0.628772; xgboost-gpu -; xgboost-cpu 0.631473; lightgbm-cpu 0.632665 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | 0.528285 | catboost-cpu 0.528923; xgboost-gpu -; xgboost-cpu 0.528686; lightgbm-cpu 0.528094 |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.983304 | catboost-cpu 0.983135; xgboost-gpu -; xgboost-cpu 0.983622 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.156072 | catboost-cpu 0.157692; xgboost-gpu -; xgboost-cpu 0.149263 |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.632205 | catboost-cpu 0.632578; xgboost-gpu -; xgboost-cpu 0.630968 |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.527838 | catboost-cpu 0.527851; xgboost-gpu -; xgboost-cpu 0.528677 |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983586 | catboost-cpu 0.983135; xgboost-gpu -; xgboost-cpu 0.983622; lightgbm-cpu 0.983778 |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.150066 | catboost-cpu 0.157692; xgboost-gpu -; xgboost-cpu 0.149263; lightgbm-cpu 0.149653 |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.632096 | catboost-cpu 0.632578; xgboost-gpu -; xgboost-cpu 0.630968; lightgbm-cpu 0.632243 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528044 | catboost-cpu 0.527851; xgboost-gpu -; xgboost-cpu 0.528677; lightgbm-cpu 0.528067 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.907556 | catboost-cpu 0.907768; xgboost-gpu -; xgboost-cpu 0.910140; lightgbm-cpu 0.910058 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.258413 | catboost-cpu 0.258286; xgboost-gpu -; xgboost-cpu 0.246803; lightgbm-cpu 0.245916 |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.599380 | catboost-cpu 0.599150; xgboost-gpu -; xgboost-cpu 0.601128; lightgbm-cpu 0.601580 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.012595 | catboost-cpu 1.012734; xgboost-gpu -; xgboost-cpu 1.005128; lightgbm-cpu 1.004282 |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.979444 | catboost-cpu 0.979221 |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.190928 | catboost-cpu 0.192114 |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.629203 | catboost-cpu 0.628918 |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.529007 | catboost-cpu 0.529083 |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.854545 | catboost-cpu 0.846328; xgboost-gpu -; xgboost-cpu 0.872796 |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.719953 | catboost-cpu 0.713361; xgboost-gpu -; xgboost-cpu 0.738397 |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.650400 | catboost-cpu 0.643611; xgboost-gpu -; xgboost-cpu 0.670093 |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-cpu 0.979899 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-cpu 0.188093 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-cpu 0.630269 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-cpu 0.528650 |


## Neural headline: ours IDENTICAL against torch bf16

What customers run is torch in bf16. The headline divides our IDENTICAL median by torch's fastest bf16 arm (eager or compile, the lower stored median); the fp32 twin (torch's fastest fp32 arm) is the second column. Per-arm ratios stay in each race's table below.

| lane | dataset | ours IDENTICAL ms | torch bf16 ms (fastest arm) | ours / torch bf16 | torch fp32 twin ms (fastest arm) | ours / torch fp32 | note |
|---|---|---|---|---|---|---|---|
| adafactor | synthetic | 8.5 | - | - | 5.8 (torch-eager-fp32) | 1.459 | no torch bf16 arm on this lane |
| adagrad | synthetic | 3.3 | - | - | 4.5 (torch-eager-fp32) | 0.728 | no torch bf16 arm on this lane |
| adamax | synthetic | 3.7 | - | - | 5.7 (torch-eager-fp32) | 0.646 | no torch bf16 arm on this lane |
| avgpool1d | synthetic | 0.5 | 0.8 (torch-eager-bf16) | 0.623 | 0.8 (torch-eager-fp32) | 0.632 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| avgpool2d | synthetic | 0.7 | 1.0 (torch-eager-bf16) | 0.730 | 0.9 (torch-eager-fp32) | 0.816 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm1d | synthetic | 2.1 | 1.0 (torch-eager-bf16) | 2.023 | 0.9 (torch-eager-fp32) | 2.269 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm2d | synthetic | 2.2 | 0.9 (torch-eager-bf16) | 2.445 | 1.0 (torch-eager-fp32) | 2.204 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| cnn-clf | synthetic | 315.0 | 258.5 (torch-eager-bf16) | 1.218 | 178.2 (torch-eager-fp32) | 1.768 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv1d | synthetic | 7.3 | 1.4 (torch-eager-bf16) | 5.340 | 1.9 (torch-eager-fp32) | 3.745 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv2d | synthetic | 5.8 | 1.4 (torch-eager-bf16) | 4.190 | 1.3 (torch-eager-fp32) | 4.470 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| dropout2d | synthetic | 0.4 | - | - | 0.9 (torch-eager-fp32) | 0.423 | no torch bf16 arm on this lane |
| embedding | synthetic | 2.0 | - | - | 1.1 (torch-compile-fp32) | 1.792 | no torch bf16 arm on this lane |
| gcn | istella | 3.0 | 3.1 (torch-compile-bf16) | 0.977 | 3.9 (torch-compile-fp32) | 0.769 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gcn | taxi | 1.4 | 2.8 (torch-compile-bf16) | 0.482 | 3.4 (torch-compile-fp32) | 0.399 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-avgpool | synthetic | 1.1 | 0.6 (torch-eager-bf16) | 1.737 | 0.7 (torch-eager-fp32) | 1.704 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-maxpool | synthetic | 1.1 | 0.6 (torch-eager-bf16) | 1.834 | 0.6 (torch-eager-fp32) | 2.033 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | istella | 5.9 | 2.6 (torch-eager-bf16) | 2.269 | 2.7 (torch-eager-fp32) | 2.199 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | taxi | 1.6 | 1.5 (torch-eager-bf16) | 1.075 | 1.5 (torch-eager-fp32) | 1.069 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | synthetic | 514.9 | 1794.7 (torch-compile-bf16) | 0.287 | 1740.4 (torch-eager-fp32) | 0.296 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | taxi-hourly | 515.5 | 1892.1 (torch-eager-bf16) | 0.272 | 1692.5 (torch-eager-fp32) | 0.305 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | synthetic | 513.3 | 1727.4 (torch-compile-bf16) | 0.297 | 1751.3 (torch-compile-fp32) | 0.293 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | taxi-hourly | 513.4 | 1727.7 (torch-compile-bf16) | 0.297 | 1724.0 (torch-compile-fp32) | 0.298 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lamb | synthetic | 13.4 | - | - | - | - | no torch bf16 arm on this lane |
| layernorm | synthetic | 1.2 | 0.9 (torch-eager-bf16) | 1.389 | 0.9 (torch-eager-fp32) | 1.376 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lion | synthetic | 3.4 | - | - | - | - | no torch bf16 arm on this lane |
| lstm-clf | synthetic | 631.6 | 936.9 (torch-compile-bf16) | 0.674 | 863.4 (torch-compile-fp32) | 0.731 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-clf | taxi-hourly | 630.7 | 962.8 (torch-compile-bf16) | 0.655 | 881.9 (torch-compile-fp32) | 0.715 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | synthetic | 628.6 | 993.6 (torch-eager-bf16) | 0.633 | 874.9 (torch-eager-fp32) | 0.718 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | taxi-hourly | 627.9 | 948.1 (torch-eager-bf16) | 0.662 | 849.2 (torch-eager-fp32) | 0.739 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool1d | synthetic | 0.4 | 0.8 (torch-eager-bf16) | 0.481 | 0.8 (torch-eager-fp32) | 0.541 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool2d | synthetic | 0.6 | 1.0 (torch-eager-bf16) | 0.673 | 0.7 (torch-eager-fp32) | 0.869 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| moe | synthetic | 15.2 | 2.7 (torch-eager-bf16) | 5.608 | 4.5 (torch-eager-fp32) | 3.412 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| nadam | synthetic | 4.0 | - | - | 7.0 (torch-eager-fp32) | 0.572 | no torch bf16 arm on this lane |
| resnet-block | synthetic | 15.5 | 2.0 (torch-eager-bf16) | 7.553 | 2.0 (torch-eager-fp32) | 7.666 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rmsprop | synthetic | 3.4 | - | - | 4.9 (torch-eager-fp32) | 0.692 | no torch bf16 arm on this lane |
| rnn-clf | synthetic | 305.9 | 804.1 (torch-compile-bf16) | 0.380 | 797.3 (torch-compile-fp32) | 0.384 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-clf | taxi-hourly | 306.6 | 932.6 (torch-eager-bf16) | 0.329 | 829.9 (torch-compile-fp32) | 0.369 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | synthetic | 300.1 | 939.1 (torch-eager-bf16) | 0.320 | 856.8 (torch-eager-fp32) | 0.350 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | taxi-hourly | 300.7 | 916.4 (torch-eager-bf16) | 0.328 | 845.1 (torch-eager-fp32) | 0.356 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gemm | gaussian | 27.7 | 12.8 (torch-compile-bf16) | 2.157 | 12.0 (torch-eager-fp32) | 2.315 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-forward | bytes | 33.6 | 13.3 (torch-compile-bf16) | 2.522 | 11.0 (torch-compile-fp32) | 3.047 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-train-step | bytes | 43.0 | 13.4 (torch-eager-bf16) | 3.204 | 10.3 (torch-compile-fp32) | 4.184 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba1-forward | gaussian | 3.1 | 110.9 (torch-eager-bf16) | 0.028 | 87.7 (torch-eager-fp32) | 0.036 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba2-forward | gaussian | 4.4 | 8.2 (torch-eager-bf16) | 0.540 | 9.0 (torch-eager-fp32) | 0.492 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mlp-train-step | gaussian | 3.4 | 1.5 (torch-eager-bf16) | 2.336 | 1.7 (torch-eager-fp32) | 1.961 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-forward | bytes | 5.6 | 4.5 (torch-compile-bf16) | 1.254 | 4.4 (torch-compile-fp32) | 1.283 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-train-step | bytes | 69.0 | 113.9 (torch-eager-bf16) | 0.606 | 30.4 (torch-compile-fp32) | 2.271 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| transformer-forward | gaussian | 3.2 | 1.5 (torch-eager-bf16) | 2.094 | 1.8 (torch-compile-fp32) | 1.747 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |

## Trees

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1694)`, ran on AMD Instinct MI325X (amd, DO) job a1694

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 10482.7 | 10482.7..10482.7 | 1 | - | - | 10482.7 | - | - (stored whole) | - | - | - | - | auc=0.631297, logloss=0.528285 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1694 2026-10-11; identity vs nvidia-l40s: MATCH) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 97857.4 | 97857.4..97857.4 | 1 | 0.107 | - | 97857.4 | 97857.4 | 0.00 (cpu-arm) | 0.107 (whole/whole) | - | 11918.4 | - | auc=0.628772, logloss=0.528923 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 18924.8 | 18924.8..18924.8 | 1 | 0.554 | - | 18924.8 | 18924.8 | 0.00 (cpu-arm) | 0.554 (whole/whole) | - | 8838.5 | - | auc=0.631473, logloss=0.528686 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 10928.0 | 10928.0..10928.0 | 1 | 0.959 | - | 10928.0 | 10928.0 | 0.00 (cpu-arm) | 0.959 (whole/whole) | - | 8741.9 | - | auc=0.632665, logloss=0.528094 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 6295.4 | 6295.4..6295.4 | 1 | - | - | 6295.4 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 70785.7 | 70785.7..70785.7 | 1 | 0.089 | - | 70785.7 | 70785.7 | 0.00 (cpu-arm) | 0.089 (whole/whole) | - | 10524.7 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 16034.8 | 16034.8..16034.8 | 1 | 0.393 | - | 16034.8 | 16034.8 | 0.00 (cpu-arm) | 0.393 (whole/whole) | - | 11639.5 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on AMD Instinct MI325X (amd, DO) job a0499

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3631.3 | 3631.3..3631.3 | 1 | - | - | 3631.3 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 43791.1 | 43791.1..43791.1 | 1 | 0.083 | - | 43791.1 | 43791.1 | 0.00 (cpu-arm) | 0.083 (whole/whole) | - | 7279.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 8079.8 | 8079.8..8079.8 | 1 | 0.449 | - | 8079.8 | 8079.8 | 0.00 (cpu-arm) | 0.449 (whole/whole) | - | 7413.2 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1694)`, ran on AMD Instinct MI325X (amd, DO) job a1694

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 6445.8 | 6445.8..6445.8 | 1 | - | - | 6445.8 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1694 2026-10-11; identity vs nvidia-l40s: MATCH) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 103179.7 | 103179.7..103179.7 | 1 | 0.062 | - | 103179.7 | 103179.7 | 0.00 (cpu-arm) | 0.062 (whole/whole) | - | 10528.0 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 23613.1 | 23613.1..23613.1 | 1 | 0.273 | - | 23613.1 | 23613.1 | 0.00 (cpu-arm) | 0.273 (whole/whole) | - | 11234.1 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 19843.3 | 19843.3..19843.3 | 1 | 0.325 | - | 19843.3 | 19843.3 | 0.00 (cpu-arm) | 0.325 (whole/whole) | - | 11308.2 | - | auc=0.983778, logloss=0.149653 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1694)`, ran on AMD Instinct MI325X (amd, DO) job a1694

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3708.8 | 3708.8..3708.8 | 1 | - | - | 3708.8 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1694 2026-10-11; identity vs nvidia-l40s: MATCH) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 77432.0 | 77432.0..77432.0 | 1 | 0.048 | - | 77432.0 | 77432.0 | 0.00 (cpu-arm) | 0.048 (whole/whole) | - | 6674.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 10350.4 | 10350.4..10350.4 | 1 | 0.358 | - | 10350.4 | 10350.4 | 0.00 (cpu-arm) | 0.358 (whole/whole) | - | 6853.6 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 10099.2 | 10099.2..10099.2 | 1 | 0.367 | - | 10099.2 | 10099.2 | 0.00 (cpu-arm) | 0.367 (whole/whole) | - | 6933.8 | - | auc=0.632243, logloss=0.528067 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 10051.4 | 10051.4..10051.4 | 1 | - | - | 10051.4 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 561543.4 | 561543.4..561543.4 | 1 | 0.018 | - | 561543.4 | 561543.4 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 11547.8 | - | accuracy=0.907768, mlogloss=0.258286 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 76053.9 | 76053.9..76053.9 | 1 | 0.132 | - | 76053.9 | 76053.9 | 0.00 (cpu-arm) | 0.132 (whole/whole) | - | 12606.4 | - | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 79078.6 | 79078.6..79078.6 | 1 | 0.127 | - | 79078.6 | 79078.6 | 0.00 (cpu-arm) | 0.127 (whole/whole) | - | 12549.6 | - | accuracy=0.910058, mlogloss=0.245916 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1116)`, ran on AMD Instinct MI325X (amd, DO) job a1116

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@d886b53dc) | identical | 6049.8 | 6049.8..6049.8 | 1 | - | - | 6049.8 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@d886b53dc amd/a1116 2026-10-09; identity vs the nvidia columns: n/a) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 178349.9 | 178349.9..178349.9 | 1 | 0.034 | - | 178349.9 | 178349.9 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 8021.3 | - | accuracy=0.599150, mlogloss=1.012734 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 35861.4 | 35861.4..35861.4 | 1 | 0.169 | - | 35861.4 | 35861.4 | 0.00 (cpu-arm) | 0.169 (whole/whole) | - | 8463.4 | - | accuracy=0.601128, mlogloss=1.005128 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | lightgbm 4.7.0 | opponent | 41539.9 | 41539.9..41539.9 | 1 | 0.146 | - | 41539.9 | 41539.9 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 8711.3 | - | accuracy=0.601580, mlogloss=1.004282 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

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
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | catboost 1.2.10 | opponent | 31176.3 | 31176.3..31176.3 | 1 | 0.049 | - | 31176.3 | 31176.3 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 12433.4 | - | map=0.846328, ndcg10=0.713361, ndcg5=0.643611 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | xgboost 3.2.0 | opponent | 6075.9 | 6075.9..6075.9 | 1 | 0.254 | - | 6075.9 | 6075.9 | 0.00 (cpu-arm) | 0.254 (whole/whole) | - | 13581.1 | - | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

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
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 54245.1 | 54245.1..54245.1 | 1 | - | - | 54245.1 | - | - (stored whole) | - | - | - | - | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1558 2026-10-10; identity vs the nvidia columns: n/a) |
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
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 7056.5 | 7056.5..7056.5 | 1 | - | - | 7056.5 | - | - (stored whole) | - | - | - | - | n_clusters=36, noise_fraction=0.000174, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 amd/a1558 2026-10-10; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | HOST-MEMORY(killed at 143.0 GB: the driver's process tree held 153.6 GB, over 90% of the box's 168.8 GB) (copied from release-board-resume-r2; measured this run) |

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1714)`, ran on AMD Instinct MI325X (amd, DO) job a1714

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 955.2 | 955.2..955.2 | 1 | - | - | 955.2 | - | - (stored whole) | - | - | - | - | n_clusters=47, noise_fraction=0.253810, rows=100000 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1714 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 829354.4 | 829354.4..829354.4 | 1 | 0.001 | - | 829354.4 | 829354.4 | 0.00 (cpu-arm) | 0.001 (whole/whole) | - | 1462.1 | - | n_clusters=52, noise_fraction=0.252570, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1714)`, ran on AMD Instinct MI325X (amd, DO) job a1714

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 323.4 | 323.4..323.4 | 1 | - | - | 323.4 | - | - (stored whole) | - | - | - | - | n_clusters=161, noise_fraction=0.140550, rows=100000 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1714 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38362.1 | 38362.1..38362.1 | 1 | 0.008 | - | 38362.1 | 38362.1 | 0.00 (cpu-arm) | 0.008 (whole/whole) | - | 334.0 | - | n_clusters=161, noise_fraction=0.134630, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kde / istella (rows full, shape 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1704)`, ran on AMD Instinct MI325X (amd, DO) job a1704

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 23.4 | 23.4..23.4 | 1 | - | - | 23.4 | - | - (stored whole) | - | - | - | - | mean_log_likelihood=-222.270583, rows_without_density=0 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1704 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 57344.0 | 57344.0..57344.0 | 1 | 0.0004081 | - | 57344.0 | 57344.0 | 0.00 (cpu-arm) | 0.0004081 (whole/whole) | - | 466.0 | - | mean_log_likelihood=-226.977407, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1704)`, ran on AMD Instinct MI325X (amd, DO) job a1704

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 12.0 | 12.0..12.0 | 1 | - | - | 12.0 | - | - (stored whole) | - | - | - | - | mean_log_likelihood=-14.826436, rows_without_density=0 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1704 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 7083.5 | 7083.5..7083.5 | 1 | 0.002 | - | 7083.5 | 7083.5 | 0.00 (cpu-arm) | 0.002 (whole/whole) | - | 209.0 | - | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1163)`, ran on AMD Instinct MI325X (amd, DO) job a1163

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 296.1 | 296.1..296.1 | 1 | - | - | 296.1 | - | - (stored whole) | - | - | - | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1882.4 | 1882.4..1882.4 | 1 | 0.157 | - | 1996.4 | 1882.4 | 114.07 (upload_ms_untimed) | 0.148 (whole/whole (kernel not derivable)) | - | 5867.6 | 3536.8 | inertia=5.991e+17, inertia_over_ours=0.990156, n_iter=57 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
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
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 187.2 | 187.2..187.2 | 1 | - | - | 187.2 | - | - (stored whole) | - | - | - | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 798.3 | 798.3..798.3 | 1 | 0.235 | - | 880.1 | 798.3 | 81.76 (upload_ms_untimed) | 0.213 (whole/whole (kernel not derivable)) | - | 3940.1 | 506.6 | inertia=3.129e+08, inertia_over_ours=1.011535, n_iter=80 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
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
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 29.1 | 29.1..29.1 | 1 | 1.034 | - | 113.6 | 29.1 | 84.57 (upload_ms_untimed) | 0.265 (whole/whole (kernel not derivable)) | - | 3225.3 | 3884.6 | recall_at_k=0.979840, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
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
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 23.9 | 23.9..23.9 | 1 | 0.555 | - | 100.8 | 23.9 | 76.94 (upload_ms_untimed) | 0.131 (whole/whole (kernel not derivable)) | - | 2903.6 | 3242.7 | recall_at_k=0.999738, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 444.7 | 444.7..444.7 | 1 | - | - | 444.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 6950.7 | 6950.7..6950.7 | 1 | 0.064 | - | 7097.9 | 6950.7 | 147.16 (upload_ms_untimed) | 0.063 (whole/whole (kernel not derivable)) | - | 3897.5 | 5304.1 | finite=False, r2=nan, rmse=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu-eigh | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 31.1 | 31.1..31.1 | 1 | 14.311 | - | 183.4 | 31.1 | 152.32 (upload_ms_untimed) | 2.425 (whole/whole (kernel not derivable)) | - | 4890.2 | 3650.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3980.6 | 3980.6..3980.6 | 1 | 0.112 | - | 3980.6 | 3980.6 | 0.00 (cpu-arm) | 0.112 (whole/whole) | - | 5793.0 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:47Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 14.1 | 14.1..14.1 | 1 | - | - | 14.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908837, rmse=4.696477 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 113.3 | 113.3..113.3 | 1 | 0.124 | - | 195.0 | 113.3 | 81.67 (upload_ms_untimed) | 0.072 (whole/whole (kernel not derivable)) | - | 1892.4 | 695.4 | finite=True, r2=0.908837, rmse=4.696472 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu-eigh | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 24.3 | 24.3..24.3 | 1 | 0.578 | - | 106.9 | 24.3 | 82.57 (upload_ms_untimed) | 0.132 (whole/whole (kernel not derivable)) | - | 2871.0 | 572.0 | finite=True, r2=0.908827, rmse=4.696715 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 304.7 | 304.7..304.7 | 1 | 0.046 | - | 304.7 | 304.7 | 0.00 (cpu-arm) | 0.046 (whole/whole) | - | 782.5 | - | finite=True, r2=0.724850, rmse=8.159187 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:14Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1676)`, ran on AMD Instinct MI325X (amd, DO) job a1676

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 51.8 | 51.8..51.8 | 1 | - | - | 51.8 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1676 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 9.3 | 9.3..9.3 | 1 | 5.578 | - | 130.5 | 9.3 | 121.20 (upload_ms_untimed) | 0.397 (whole/whole (kernel not derivable)) | - | 4861.4 | 3506.7 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1676)`, ran on AMD Instinct MI325X (amd, DO) job a1676

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 7.6 | 7.6..7.6 | 1 | - | - | 7.6 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999997 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1676 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 6.574 | - | 82.9 | 1.2 | 81.75 (upload_ms_untimed) | 0.091 (whole/whole (kernel not derivable)) | - | 2908.1 | 412.0 | explained_variance_ratio_sum=0.999996 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 82.2 | 82.2..82.2 | 1 | 0.092 | - | 82.2 | 82.2 | 0.00 (cpu-arm) | 0.092 (whole/whole) | - | 401.6 | - | explained_variance_ratio_sum=0.999995 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:34:39Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svc / istella (rows full, shape 10000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1708)`, ran on AMD Instinct MI325X (amd, DO) job a1708

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 51.7 | 51.7..51.7 | 1 | - | - | 51.7 | - | - (stored whole) | - | - | - | - | accuracy=0.922200, n_support=2400 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1708 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 854.5 | 854.5..854.5 | 1 | 0.060 | - | 854.5 | 854.5 | 0.00 (cpu-arm) | 0.060 (whole/whole) | - | 335.4 | - | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1708)`, ran on AMD Instinct MI325X (amd, DO) job a1708

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1511.4 | 1511.4..1511.4 | 1 | - | - | 1511.4 | - | - (stored whole) | - | - | - | - | accuracy=0.767500, n_support=5527 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1708 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 1727.5 | 1727.5..1727.5 | 1 | 0.875 | - | 1727.5 | 1727.5 | 0.00 (cpu-arm) | 0.875 (whole/whole) | - | 455.3 | - | accuracy=0.767500, n_support=5675 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

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
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 194.9 | 194.9..194.9 | 1 | - | - | 194.9 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999925, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs the nvidia columns: n/a) |
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
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 44.7 | 44.7..44.7 | 1 | - | - | 44.7 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999650, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1163 2026-10-09; identity vs the nvidia columns: n/a) |
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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 419.7 | 419.7..419.7 | 1 | - | - | 419.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328674, rmse=0.684427 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 4288.8 | 4288.8..4288.8 | 1 | 0.098 | - | 4288.8 | 4288.8 | 0.00 (cpu-arm) | 0.098 (whole/whole) | - | 8688.9 | - | finite=True, r2=0.328676, rmse=0.684426 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1716)`, ran on AMD Instinct MI325X (amd, DO) job a1716

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1716 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 33.2 | 33.2..33.2 | 1 | 0.103 | - | 33.2 | 33.2 | 0.00 (cpu-arm) | 0.103 (whole/whole) | - | 291.3 | - | finite=True, r2=0.908983, rmse=4.805056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1676)`, ran on AMD Instinct MI325X (amd, DO) job a1676

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 29.6 | 29.6..29.6 | 1 | - | - | 29.6 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001314 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1676 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 689.8 | 689.8..689.8 | 1 | 0.043 | - | 689.8 | 689.8 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 2422.2 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.000122 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 12.8 ms) = 2.157; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 12.0 ms) = 2.315. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 27.7 | 27.7..27.7 | 1 | - | - | 27.7 | - | - (stored whole) | - | - | - | - | max_rel_err_vs_fp64=2.399e-07 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 12.0 | 12.0..12.0 | 1 | 2.315 | - | 12.0 | - | - (stored whole) | 2.315 (whole/whole (kernel not derivable)) | - | 1789.3 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 14.5 | 14.5..14.5 | 1 | 1.914 | - | 14.5 | - | - (stored whole) | 1.914 (whole/whole (kernel not derivable)) | - | 2017.8 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 12.9 | 12.9..12.9 | 1 | 2.141 | - | 12.9 | - | - (stored whole) | 2.141 (whole/whole (kernel not derivable)) | - | 2626.3 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 12.8 | 12.8..12.8 | 1 | 2.157 | - | 12.8 | - | - (stored whole) | 2.157 (whole/whole (kernel not derivable)) | - | 3071.1 | 376.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 13.3 ms) = 2.522; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 11.0 ms) = 3.047. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 33.6 | 33.6..33.6 | 1 | - | - | 33.6 | - | - (stored whole) | - | - | - | - | mean_nll=9.018733 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 11.3 | 11.3..11.3 | 1 | 2.985 | - | 11.3 | - | - (stored whole) | 2.985 (whole/whole (kernel not derivable)) | - | 3072.9 | 241.0 | max_abs_diff_vs_ours=1.222e-06, max_rel_diff_vs_ours=1.677e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 11.0 | 11.0..11.0 | 1 | 3.047 | - | 11.0 | - | - (stored whole) | 3.047 (whole/whole (kernel not derivable)) | - | 3255.9 | 222.5 | max_abs_diff_vs_ours=1.192e-06, max_rel_diff_vs_ours=1.637e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 13.5 | 13.5..13.5 | 1 | 2.486 | - | 13.5 | - | - (stored whole) | 2.486 (whole/whole (kernel not derivable)) | - | 2797.4 | 327.5 | max_abs_diff_vs_ours=0.006313, max_rel_diff_vs_ours=0.008666, mean_nll=9.018647 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 13.3 | 13.3..13.3 | 1 | 2.522 | - | 13.3 | - | - (stored whole) | 2.522 (whole/whole (kernel not derivable)) | - | 3011.1 | 327.5 | max_abs_diff_vs_ours=0.006870, max_rel_diff_vs_ours=0.009432, mean_nll=9.018653 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 13.4 ms) = 3.204; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 10.3 ms) = 4.184. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 43.0 | 43.0..43.0 | 1 | - | - | 43.0 | - | - (stored whole) | - | - | - | - | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 12.8 | 12.8..12.8 | 1 | 3.371 | - | 12.8 | - | - (stored whole) | 3.371 (whole/whole (kernel not derivable)) | - | 4803.5 | 1089.6 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018732, loss_last_abs_diff_vs_ours=0.000000, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 10.3 | 10.3..10.3 | 1 | 4.184 | - | 10.3 | - | - (stored whole) | 4.184 (whole/whole (kernel not derivable)) | - | 4852.1 | 859.6 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018734, loss_last_abs_diff_vs_ours=9.537e-07, loss_last_step=8.422412, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 13.4 | 13.4..13.4 | 1 | 3.204 | - | 13.4 | - | - (stored whole) | 3.204 (whole/whole (kernel not derivable)) | - | 3061.0 | 930.6 | loss_first_abs_diff_vs_ours=5.627e-05, loss_first_step=9.018677, loss_last_abs_diff_vs_ours=0.002001, loss_last_step=8.420410, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 45.7 | 45.7..45.7 | 1 | 0.940 | - | 45.7 | - | - (stored whole) | 0.940 (whole/whole (kernel not derivable)) | - | 3070.1 | 688.1 | loss_first_abs_diff_vs_ours=8.011e-05, loss_first_step=9.018653, loss_last_abs_diff_vs_ours=0.003033, loss_last_step=8.419378, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1652)`, ran on AMD Instinct MI325X (amd, DO) job a1652

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 110.9 ms) = 0.028; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 87.7 ms) = 0.036. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 3.1 | 3.1..3.1 | 1 | - | - | 3.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1652 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 87.7 | 87.7..87.7 | 1 | 0.036 | - | 87.7 | - | - (stored whole) | 0.036 (whole/whole (kernel not derivable)) | - | 3964.5 | 422.2 | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.189e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 110.9 | 110.9..110.9 | 1 | 0.028 | - | 110.9 | - | - (stored whole) | 0.028 (whole/whole (kernel not derivable)) | - | 3654.8 | 442.5 | max_abs_diff_vs_ours=5.436e-05, max_rel_diff_vs_ours=2.711e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 8.2 ms) = 0.540; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 9.0 ms) = 0.492. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 4.4 | 4.4..4.4 | 1 | - | - | 4.4 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 9.0 | 9.0..9.0 | 1 | 0.492 | - | 9.0 | - | - (stored whole) | 0.492 (whole/whole (kernel not derivable)) | - | 3394.8 | 3296.3 | max_abs_diff_vs_ours=1.073e-06, max_rel_diff_vs_ours=3.648e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 10.5 | 10.5..10.5 | 1 | 0.420 | - | 10.5 | - | - (stored whole) | 0.420 (whole/whole (kernel not derivable)) | - | 3709.4 | 3296.5 | max_abs_diff_vs_ours=1.073e-06, max_rel_diff_vs_ours=3.648e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 8.2 | 8.2..8.2 | 1 | 0.540 | - | 8.2 | - | - (stored whole) | 0.540 (whole/whole (kernel not derivable)) | - | 3174.5 | 1828.8 | max_abs_diff_vs_ours=0.007144, max_rel_diff_vs_ours=0.002429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 47.0 | 47.0..47.0 | 1 | 0.094 | - | 47.0 | - | - (stored whole) | 0.094 (whole/whole (kernel not derivable)) | - | 3467.4 | 1828.8 | max_abs_diff_vs_ours=0.007144, max_rel_diff_vs_ours=0.002429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 2.336; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.7 ms) = 1.961. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.7 | 1.7..1.7 | 1 | 1.961 | - | 1.7 | - | - (stored whole) | 1.961 (whole/whole (kernel not derivable)) | - | 4354.9 | 152.1 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.8 | 1.8..1.8 | 1 | 1.925 | - | 1.8 | - | - (stored whole) | 1.925 (whole/whole (kernel not derivable)) | - | 4263.6 | 152.1 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 2.336 | - | 1.5 | - | - (stored whole) | 2.336 (whole/whole (kernel not derivable)) | - | 2707.7 | 152.0 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001005, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 71.3 | 71.3..71.3 | 1 | 0.048 | - | 71.3 | - | - (stored whole) | 0.048 (whole/whole (kernel not derivable)) | - | 2828.7 | 152.0 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a1189)`, ran on AMD Instinct MI325X (amd, DO) job a1189

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 4.5 ms) = 1.254; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 4.4 ms) = 1.283. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 5.6 | 5.6..5.6 | 1 | - | - | 5.6 | - | - (stored whole) | - | - | - | - | mean_nll=5.635910 | - | main board, one scored run | - | ok (main@9f83ea479 amd/a1189 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 40.5 | 40.5..40.5 | 1 | 0.139 | - | 40.5 | - | - (stored whole) | 0.139 (whole/whole (kernel not derivable)) | - | 3922.7 | 205.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=1.206e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.4 | 4.4..4.4 | 1 | 1.283 | - | 4.4 | - | - (stored whole) | 1.283 (whole/whole (kernel not derivable)) | - | 4862.8 | 132.8 | max_abs_diff_vs_ours=2.168e-06, max_rel_diff_vs_ours=1.219e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 40.3 | 40.3..40.3 | 1 | 0.140 | - | 40.3 | - | - (stored whole) | 0.140 (whole/whole (kernel not derivable)) | - | 3343.2 | 280.2 | max_abs_diff_vs_ours=0.017180, max_rel_diff_vs_ours=0.009659, mean_nll=5.635950 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.5 | 4.5..4.5 | 1 | 1.254 | - | 4.5 | - | - (stored whole) | 1.254 (whole/whole (kernel not derivable)) | - | 4269.0 | 209.3 | max_abs_diff_vs_ours=0.016678, max_rel_diff_vs_ours=0.009377, mean_nll=5.635933 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1649)`, ran on AMD Instinct MI325X (amd, DO) job a1649

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 113.9 ms) = 0.606; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 30.4 ms) = 2.271. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 69.0 | 69.0..69.0 | 1 | - | - | 69.0 | - | - (stored whole) | - | - | - | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1649 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 116.1 | 116.1..116.1 | 1 | 0.595 | - | 116.1 | - | - (stored whole) | 0.595 (whole/whole (kernel not derivable)) | - | 5209.4 | 534.3 | loss_first_abs_diff_vs_ours=4.768e-07, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 30.4 | 30.4..30.4 | 1 | 2.271 | - | 30.4 | - | - (stored whole) | 2.271 (whole/whole (kernel not derivable)) | - | 7777.2 | 449.6 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=5.635909, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 113.9 | 113.9..113.9 | 1 | 0.606 | - | 113.9 | - | - (stored whole) | 0.606 (whole/whole (kernel not derivable)) | - | 3500.1 | 485.2 | loss_first_abs_diff_vs_ours=3.91e-05, loss_first_step=5.635949, loss_last_abs_diff_vs_ours=8.106e-05, loss_last_step=4.834015, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on cuda failed in round 0 (compile happens here): InductorError('RuntimeError: A compilation subprocess exited unexpectedly. This is likely due to a crash. To fa) (copied from release-board-resume-r2; measured this run) |

memory, ours, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-grid-logs.txt (amd/a1697)`, ran on AMD Instinct MI325X (amd, DO) job a1697

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 2.094; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.8 ms) = 1.747. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1697 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.9 | 1.9..1.9 | 1 | 1.676 | - | 1.9 | - | - (stored whole) | 1.676 (whole/whole (kernel not derivable)) | - | 2827.0 | 147.3 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.8 | 1.8..1.8 | 1 | 1.747 | - | 1.8 | - | - (stored whole) | 1.747 (whole/whole (kernel not derivable)) | - | 3015.3 | 119.3 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 2.094 | - | 1.5 | - | - (stored whole) | 2.094 (whole/whole (kernel not derivable)) | - | 2661.9 | 215.1 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 65.8 | 65.8..65.8 | 1 | 0.049 | - | 65.8 | - | - (stored whole) | 0.049 (whole/whole (kernel not derivable)) | - | 2808.3 | 178.3 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1194)`, ran on AMD Instinct MI325X (amd, DO) job a1194

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.8 ms) = 1.459. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@8d5855b83) | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@8d5855b83 amd/a1194 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 5.8 | 5.8..5.8 | 1 | 1.459 | - | - | 5.8 | - (stored kernel) | 1.459 (MIXED ours whole / arm kernel) | - | 2717.2 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 102.6 | 102.6..102.6 | 1 | 0.083 | - | - | 102.6 | - (stored kernel) | 0.083 (MIXED ours whole / arm kernel) | - | 2807.9 | 1024.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adagrad / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 4.5 ms) = 0.728. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 3.3 | 3.3..3.3 | 1 | - | - | 3.3 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.5 | 4.5..4.5 | 1 | 0.728 | - | - | 4.5 | - (stored kernel) | 0.728 (MIXED ours whole / arm kernel) | - | 2621.6 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 51.2 | 51.2..51.2 | 1 | 0.064 | - | - | 51.2 | - (stored kernel) | 0.064 (MIXED ours whole / arm kernel) | - | 3014.9 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 5.7 ms) = 0.646. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 3.7 | 3.7..3.7 | 1 | - | - | 3.7 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 5.7 | 5.7..5.7 | 1 | 0.646 | - | - | 5.7 | - (stored kernel) | 0.646 (MIXED ours whole / arm kernel) | - | 2631.7 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 218.6 | 218.6..218.6 | 1 | 0.017 | - | - | 218.6 | - (stored kernel) | 0.017 (MIXED ours whole / arm kernel) | - | 3145.7 | 896.0 | rel_fro_vs_torch_eager_fp32=6.17e-09 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ard / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1661)`, ran on AMD Instinct MI325X (amd, DO) job a1661

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 1234.1 | 1234.1..1234.1 | 1 | - | - | 1234.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.124856, rmse=0.885949 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1661 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 14123.5 | 14123.5..14123.5 | 1 | 0.087 | - | 14123.5 | 14123.5 | 0.00 (cpu-arm) | 0.087 (whole/whole) | - | 1389.6 | - | finite=True, r2=0.327436, rmse=0.685058 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.8 ms) = 0.623; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.8 ms) = 0.632. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.632 | - | - | 0.8 | - (stored kernel) | 0.632 (MIXED ours whole / arm kernel) | - | 1599.6 | 224.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 0.502 | - | - | 1.0 | - (stored kernel) | 0.502 (MIXED ours whole / arm kernel) | - | 2050.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.623 | - | - | 0.8 | - (stored kernel) | 0.623 (MIXED ours whole / arm kernel) | - | 1599.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.1 | 1.1..1.1 | 1 | 0.430 | - | - | 1.1 | - (stored kernel) | 0.430 (MIXED ours whole / arm kernel) | - | 1965.6 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1664)`, ran on AMD Instinct MI325X (amd, DO) job a1664

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.0 ms) = 0.730; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.9 ms) = 0.816. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 0.7 | 0.7..0.7 | 1 | - | - | 0.7 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1664 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.816 | - | - | 0.9 | - (stored kernel) | 0.816 (MIXED ours whole / arm kernel) | - | 1790.3 | 541.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 0.563 | - | - | 1.3 | - (stored kernel) | 0.563 (MIXED ours whole / arm kernel) | - | 2202.2 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 0.730 | - | - | 1.0 | - (stored kernel) | 0.730 (MIXED ours whole / arm kernel) | - | 1790.0 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 0.559 | - | - | 1.3 | - (stored kernel) | 0.559 (MIXED ours whole / arm kernel) | - | 2121.6 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1721)`, ran on AMD Instinct MI325X (amd, DO) job a1721

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.0 ms) = 2.023; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.9 ms) = 2.269. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@7924718fc) | identical | 2.1 | 2.1..2.1 | 1 | - | - | 2.1 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@7924718fc amd/a1721 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 2.269 | - | - | 0.9 | - (stored kernel) | 2.269 (MIXED ours whole / arm kernel) | - | 2260.1 | 320.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 1.406 | - | - | 1.5 | - (stored kernel) | 1.406 (MIXED ours whole / arm kernel) | - | 2114.8 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 2.023 | - | - | 1.0 | - (stored kernel) | 2.023 (MIXED ours whole / arm kernel) | - | 1892.7 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 1.538 | - | - | 1.4 | - (stored kernel) | 1.538 (MIXED ours whole / arm kernel) | - | 2008.9 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002793, rel_fro_vs_torch_eager_fp32=5.248e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1721)`, ran on AMD Instinct MI325X (amd, DO) job a1721

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.9 ms) = 2.445; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.0 ms) = 2.204. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@7924718fc) | identical | 2.2 | 2.2..2.2 | 1 | - | - | 2.2 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@7924718fc amd/a1721 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 2.204 | - | - | 1.0 | - (stored kernel) | 2.204 (MIXED ours whole / arm kernel) | - | 2193.7 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 1.630 | - | - | 1.4 | - (stored kernel) | 1.630 (MIXED ours whole / arm kernel) | - | 2083.4 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 2.445 | - | - | 0.9 | - (stored kernel) | 2.445 (MIXED ours whole / arm kernel) | - | 1827.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 1.607 | - | - | 1.4 | - (stored kernel) | 1.607 (MIXED ours whole / arm kernel) | - | 1977.6 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001471, rel_fro_vs_torch_eager_fp32=6.868e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bayesian-ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1661)`, ran on AMD Instinct MI325X (amd, DO) job a1661

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 1130.2 | 1130.2..1130.2 | 1 | - | - | 1130.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.317909, rmse=0.689893 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1661 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 3401.4 | 3401.4..3401.4 | 1 | 0.332 | - | 3401.4 | 3401.4 | 0.00 (cpu-arm) | 0.332 (whole/whole) | - | 4508.1 | - | finite=True, r2=-8.145e+07, rmse=7538.692925 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1592)`, ran on AMD Instinct MI325X (amd, DO) job a1592

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 157.4 | 157.4..157.4 | 1 | - | - | 157.4 | - | - (stored whole) | - | - | - | - | relative_residual=2.9e-07 | - | main board, one scored run | - | ok (main@5876e9205 amd/a1592 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 7138.9 | 7138.9..7138.9 | 1 | 0.022 | - | 7138.9 | 7138.9 | 0.00 (cpu-arm) | 0.022 (whole/whole) | - | 2650.2 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 32.0 | 32.0..32.0 | 1 | 4.924 | - | - | 32.0 | - (stored kernel) | 4.924 (MIXED ours whole / arm kernel) | - | 2616.0 | 790.4 | relative_residual=2.519e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1647)`, ran on AMD Instinct MI325X (amd, DO) job a1647

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 258.5 ms) = 1.218; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 178.2 ms) = 1.768. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 315.0 | 315.0..315.0 | 1 | - | - | 315.0 | - | - (stored whole) | - | - | - | - | accuracy=1.000000 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1647 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 178.2 | 178.2..178.2 | 1 | 1.768 | - | - | 178.2 | - (stored kernel) | 1.768 (MIXED ours whole / arm kernel) | - | 5759.4 | 423.5 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 210.1 | 210.1..210.1 | 1 | 1.499 | - | - | 210.1 | - (stored kernel) | 1.499 (MIXED ours whole / arm kernel) | - | 5130.7 | 350.0 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 258.5 | 258.5..258.5 | 1 | 1.218 | - | - | 258.5 | - (stored kernel) | 1.218 (MIXED ours whole / arm kernel) | - | 4287.9 | 431.6 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 287.6 | 287.6..287.6 | 1 | 1.095 | - | - | 287.6 | - (stored kernel) | 1.095 (MIXED ours whole / arm kernel) | - | 3433.3 | 431.6 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 67.1 | 67.1..67.1 | 1 | - | - | 67.1 | - | - (stored whole) | - | - | - | - | accuracy=0.849360, logloss=3.762524 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 196.8 | 196.8..196.8 | 1 | 0.341 | - | 196.8 | 196.8 | 0.00 (cpu-arm) | 0.341 (whole/whole) | - | 3958.9 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 29.3 | 29.3..29.3 | 1 | - | - | 29.3 | - | - (stored whole) | - | - | - | - | accuracy=0.678020, logloss=0.715492 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38.2 | 38.2..38.2 | 1 | 0.769 | - | 38.2 | 38.2 | 0.00 (cpu-arm) | 0.769 (whole/whole) | - | 450.3 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1664)`, ran on AMD Instinct MI325X (amd, DO) job a1664

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.4 ms) = 5.340; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.9 ms) = 3.745. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 7.3 | 7.3..7.3 | 1 | - | - | 7.3 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1664/work-conv1d-synthetic-def/conv1d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1664 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.9 | 1.9..1.9 | 1 | 3.745 | - | - | 1.9 | - (stored kernel) | 3.745 (MIXED ours whole / arm kernel) | - | 2747.7 | 896.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.4 | 2.4..2.4 | 1 | 2.974 | - | - | 2.4 | - (stored kernel) | 2.974 (MIXED ours whole / arm kernel) | - | 2437.2 | 896.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 5.340 | - | - | 1.4 | - (stored kernel) | 5.340 (MIXED ours whole / arm kernel) | - | 2994.4 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.0 | 2.0..2.0 | 1 | 3.554 | - | - | 2.0 | - (stored kernel) | 3.554 (MIXED ours whole / arm kernel) | - | 2478.6 | 832.4 | max_rel_diff_vs_torch_eager_fp32=3418.035805, rel_fro_vs_torch_eager_fp32=0.004377 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1664)`, ran on AMD Instinct MI325X (amd, DO) job a1664

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.4 ms) = 4.190; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.3 ms) = 4.470. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 5.8 | 5.8..5.8 | 1 | - | - | 5.8 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1664/work-conv2d-synthetic-def/conv2d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1664 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 4.470 | - | - | 1.3 | - (stored kernel) | 4.470 (MIXED ours whole / arm kernel) | - | 2396.5 | 347.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.8 | 1.8..1.8 | 1 | 3.231 | - | - | 1.8 | - (stored kernel) | 3.231 (MIXED ours whole / arm kernel) | - | 2278.0 | 347.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 4.190 | - | - | 1.4 | - (stored kernel) | 4.190 (MIXED ours whole / arm kernel) | - | 2428.6 | 320.8 | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.6 | 1.6..1.6 | 1 | 3.520 | - | - | 1.6 | - (stored kernel) | 3.520 (MIXED ours whole / arm kernel) | - | 2329.5 | 320.8 | max_rel_diff_vs_torch_eager_fp32=3418.486565, rel_fro_vs_torch_eager_fp32=0.003382 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 1680.4 | 1680.4..1680.4 | 1 | - | - | 1680.4 | - | - (stored whole) | - | - | - | - | forecast_rmse=13.931153 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 94.5 | 94.5..94.5 | 1 | 17.790 | - | 94.5 | 94.5 | 0.00 (cpu-arm) | 17.790 (whole/whole) | - | 58.1 | - | forecast_rmse=26.588738 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 93.0 | 93.0..93.0 | 1 | 18.063 | - | 93.0 | 93.0 | 0.00 (cpu-arm) | 18.063 (whole/whole) | - | 283.5 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 1772.4 | 1772.4..1772.4 | 1 | - | - | 1772.4 | - | - (stored whole) | - | - | - | - | forecast_rmse=96.690449 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 94.0 | 94.0..94.0 | 1 | 18.847 | - | 94.0 | 94.0 | 0.00 (cpu-arm) | 18.847 (whole/whole) | - | 57.9 | - | forecast_rmse=196.955273 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 63.3 | 63.3..63.3 | 1 | 27.996 | - | 63.3 | 63.3 | 0.00 (cpu-arm) | 27.996 (whole/whole) | - | 283.6 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1156)`, ran on AMD Instinct MI325X (amd, DO) job a1156

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.9 ms) = 0.423. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 0.4 | 0.4..0.4 | 1 | - | - | 0.4 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 0.423 | - | - | 0.9 | - (stored kernel) | 0.423 (MIXED ours whole / arm kernel) | - | 1596.4 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 97.2 | 97.2..97.2 | 1 | 0.004 | - | - | 97.2 | - (stored kernel) | 0.004 (MIXED ours whole / arm kernel) | - | 1833.4 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dynamic-optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 765.3 | 765.3..765.3 | 1 | - | - | 765.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436278 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 354.2 | 354.2..354.2 | 1 | 2.161 | - | 354.2 | 354.2 | 0.00 (cpu-arm) | 2.161 (whole/whole) | - | 283.6 | - | forecast_rmse=1.436045 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dynamic-optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 4411.8 | 4411.8..4411.8 | 1 | - | - | 4411.8 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.086628 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 798.7 | 798.7..798.7 | 1 | 5.524 | - | 798.7 | 798.7 | 0.00 (cpu-arm) | 5.524 (whole/whole) | - | 283.8 | - | forecast_rmse=49.314797 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1633)`, ran on AMD Instinct MI325X (amd, DO) job a1633

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 355.8 | 355.8..355.8 | 1 | - | - | 355.8 | - | - (stored whole) | - | - | - | - | max_eigenvalue_error=3.692e-07, relative_residual=6.672e-07 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1633 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 2356.5 | 2356.5..2356.5 | 1 | 0.151 | - | 2356.5 | 2356.5 | 0.00 (cpu-arm) | 0.151 (whole/whole) | - | 970.0 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 117.5 | 117.5..117.5 | 1 | 3.028 | - | - | 117.5 | - (stored kernel) | 3.028 (MIXED ours whole / arm kernel) | - | 2037.8 | 400.7 | max_eigenvalue_error=1.044e-06, relative_residual=9.614e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### embedding / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.1 ms) = 1.792. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 2.0 | 2.0..2.0 | 1 | - | - | 2.0 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.1 | 1.1..1.1 | 1 | 1.769 | - | - | 1.1 | - (stored kernel) | 1.769 (MIXED ours whole / arm kernel) | - | 2290.0 | 782.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.1 | 1.1..1.1 | 1 | 1.792 | - | - | 1.1 | - (stored kernel) | 1.792 (MIXED ours whole / arm kernel) | - | 2342.8 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 854.8 | 854.8..854.8 | 1 | - | - | 854.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.326805, rmse=0.685379 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 47198.6 | 47198.6..47198.6 | 1 | 0.018 | - | 47198.6 | 47198.6 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 8285.3 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 222.8 | 222.8..222.8 | 1 | - | - | 222.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909004, rmse=4.804486 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 279.5 | 279.5..279.5 | 1 | 0.797 | - | 279.5 | 279.5 | 0.00 (cpu-arm) | 0.797 (whole/whole) | - | 673.5 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.1 ms) = 0.977; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.9 ms) = 0.769. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-gcn-istella-def/gcn-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.8 | 4.8..4.8 | 1 | 0.636 | - | - | 4.8 | - (stored kernel) | 0.636 (MIXED ours whole / arm kernel) | - | 4551.2 | 2070.3 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.9 | 3.9..3.9 | 1 | 0.769 | - | - | 3.9 | - (stored kernel) | 0.769 (MIXED ours whole / arm kernel) | - | 4591.6 | 535.2 | max_rel_diff_vs_torch_eager_fp32=0.007740, rel_fro_vs_torch_eager_fp32=9.044e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.9 | 4.9..4.9 | 1 | 0.617 | - | - | 4.9 | - (stored kernel) | 0.617 (MIXED ours whole / arm kernel) | - | 3211.8 | 2459.5 | max_rel_diff_vs_torch_eager_fp32=2866.979199, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.1 | 3.1..3.1 | 1 | 0.977 | - | - | 3.1 | - (stored kernel) | 0.977 (MIXED ours whole / arm kernel) | - | 3528.1 | 918.2 | max_rel_diff_vs_torch_eager_fp32=2866.972319, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 2.8 ms) = 0.482; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.4 ms) = 0.399. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 1.4 | 1.4..1.4 | 1 | - | - | 1.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-gcn-taxi-def/gcn-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.2 | 4.2..4.2 | 1 | 0.330 | - | - | 4.2 | - (stored kernel) | 0.330 (MIXED ours whole / arm kernel) | - | 4465.7 | 1725.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.4 | 3.4..3.4 | 1 | 0.399 | - | - | 3.4 | - (stored kernel) | 0.399 (MIXED ours whole / arm kernel) | - | 5152.1 | 446.3 | max_rel_diff_vs_torch_eager_fp32=0.002794, rel_fro_vs_torch_eager_fp32=7.648e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.1 | 4.1..4.1 | 1 | 0.336 | - | - | 4.1 | - (stored kernel) | 0.336 (MIXED ours whole / arm kernel) | - | 3125.5 | 2010.7 | max_rel_diff_vs_torch_eager_fp32=1509.504160, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.8 | 2.8..2.8 | 1 | 0.482 | - | - | 2.8 | - (stored kernel) | 0.482 (MIXED ours whole / arm kernel) | - | 3444.2 | 726.3 | max_rel_diff_vs_torch_eager_fp32=1509.503229, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1664)`, ran on AMD Instinct MI325X (amd, DO) job a1664

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.6 ms) = 1.737; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 1.704. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1664/work-global-avgpool-synthetic-def/global-avgpool-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1664 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 1.704 | - | - | 0.7 | - (stored kernel) | 1.704 (MIXED ours whole / arm kernel) | - | 1634.8 | 12.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 1.125 | - | - | 1.0 | - (stored kernel) | 1.125 (MIXED ours whole / arm kernel) | - | 1965.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.6 | 0.6..0.6 | 1 | 1.737 | - | - | 0.6 | - (stored kernel) | 1.737 (MIXED ours whole / arm kernel) | - | 1634.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 1.119 | - | - | 1.0 | - (stored kernel) | 1.119 (MIXED ours whole / arm kernel) | - | 1883.3 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=1.01e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1664)`, ran on AMD Instinct MI325X (amd, DO) job a1664

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.6 ms) = 1.834; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.6 ms) = 2.033. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1664 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.6 | 0.6..0.6 | 1 | 2.033 | - | - | 0.6 | - (stored kernel) | 2.033 (MIXED ours whole / arm kernel) | - | 1610.5 | 12.9 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.1 | 1.1..1.1 | 1 | 1.009 | - | - | 1.1 | - (stored kernel) | 1.009 (MIXED ours whole / arm kernel) | - | 1993.0 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.6 | 0.6..0.6 | 1 | 1.834 | - | - | 0.6 | - (stored kernel) | 1.834 (MIXED ours whole / arm kernel) | - | 1610.4 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 1.375 | - | - | 0.8 | - (stored kernel) | 1.375 (MIXED ours whole / arm kernel) | - | 1902.0 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.6 ms) = 2.269; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.7 ms) = 2.199. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 5.9 | 5.9..5.9 | 1 | - | - | 5.9 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-graphsage-istella-def/graphsage-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.7 | 2.7..2.7 | 1 | 2.199 | - | - | 2.7 | - (stored kernel) | 2.199 (MIXED ours whole / arm kernel) | - | 4278.1 | 1803.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.1 | 3.1..3.1 | 1 | 1.916 | - | - | 3.1 | - (stored kernel) | 1.916 (MIXED ours whole / arm kernel) | - | 4493.5 | 539.5 | max_rel_diff_vs_torch_eager_fp32=0.108108, rel_fro_vs_torch_eager_fp32=8.065e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.6 | 2.6..2.6 | 1 | 2.269 | - | - | 2.6 | - (stored kernel) | 2.269 (MIXED ours whole / arm kernel) | - | 2936.5 | 1778.7 | max_rel_diff_vs_torch_eager_fp32=3907.203674, rel_fro_vs_torch_eager_fp32=0.003366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.0 | 3.0..3.0 | 1 | 1.952 | - | - | 3.0 | - (stored kernel) | 1.952 (MIXED ours whole / arm kernel) | - | 3382.6 | 466.3 | max_rel_diff_vs_torch_eager_fp32=3678.128123, rel_fro_vs_torch_eager_fp32=0.003054 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1210)`, ran on AMD Instinct MI325X (amd, DO) job a1210

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 1.075; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.5 ms) = 1.069. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@6416999f7) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1210/work-graphsage-taxi-def/graphsage-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@6416999f7 amd/a1210 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 1.069 | - | - | 1.5 | - (stored kernel) | 1.069 (MIXED ours whole / arm kernel) | - | 4191.6 | 424.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.7 | 1.7..1.7 | 1 | 0.944 | - | - | 1.7 | - (stored kernel) | 0.944 (MIXED ours whole / arm kernel) | - | 4852.5 | 375.7 | max_rel_diff_vs_torch_eager_fp32=0.074506, rel_fro_vs_torch_eager_fp32=4.566e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.5 | 1.5..1.5 | 1 | 1.075 | - | - | 1.5 | - (stored kernel) | 1.075 (MIXED ours whole / arm kernel) | - | 2850.0 | 327.9 | max_rel_diff_vs_torch_eager_fp32=5040.230769, rel_fro_vs_torch_eager_fp32=0.003557 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.0 | 2.0..2.0 | 1 | 0.809 | - | - | 2.0 | - (stored kernel) | 0.809 (MIXED ours whole / arm kernel) | - | 3299.8 | 302.5 | max_rel_diff_vs_torch_eager_fp32=4607.677460, rel_fro_vs_torch_eager_fp32=0.003285 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1794.7 ms) = 0.287; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1740.4 ms) = 0.296. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 514.9 | 514.9..514.9 | 1 | - | - | 514.9 | - | - (stored whole) | - | - | - | - | accuracy=0.971625, logloss=0.070054 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1740.4 | 1740.4..1740.4 | 1 | 0.296 | - | - | 1740.4 | - (stored kernel) | 0.296 (MIXED ours whole / arm kernel) | - | 4839.2 | 476.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1795.5 | 1795.5..1795.5 | 1 | 0.287 | - | - | 1795.5 | - (stored kernel) | 0.287 (MIXED ours whole / arm kernel) | - | 4886.9 | 476.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1839.7 | 1839.7..1839.7 | 1 | 0.280 | - | - | 1839.7 | - (stored kernel) | 0.280 (MIXED ours whole / arm kernel) | - | 3216.3 | 320.7 | accuracy=0.971680 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1794.7 | 1794.7..1794.7 | 1 | 0.287 | - | - | 1794.7 | - (stored kernel) | 0.287 (MIXED ours whole / arm kernel) | - | 3265.1 | 320.7 | accuracy=0.971680 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1892.1 ms) = 0.272; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1692.5 ms) = 0.305. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 515.5 | 515.5..515.5 | 1 | - | - | 515.5 | - | - (stored whole) | - | - | - | - | accuracy=0.865723, logloss=0.303537 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1692.5 | 1692.5..1692.5 | 1 | 0.305 | - | - | 1692.5 | - (stored kernel) | 0.305 (MIXED ours whole / arm kernel) | - | 5229.5 | 476.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1798.4 | 1798.4..1798.4 | 1 | 0.287 | - | - | 1798.4 | - (stored kernel) | 0.287 (MIXED ours whole / arm kernel) | - | 4887.1 | 476.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1892.1 | 1892.1..1892.1 | 1 | 0.272 | - | - | 1892.1 | - (stored kernel) | 0.272 (MIXED ours whole / arm kernel) | - | 3667.0 | 320.7 | accuracy=0.865777 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1898.9 | 1898.9..1898.9 | 1 | 0.271 | - | - | 1898.9 | - (stored kernel) | 0.271 (MIXED ours whole / arm kernel) | - | 3266.8 | 320.7 | accuracy=0.865777 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1727.4 ms) = 0.297; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1751.3 ms) = 0.293. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 513.3 | 513.3..513.3 | 1 | - | - | 513.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.982337, rmse=0.153975 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1801.3 | 1801.3..1801.3 | 1 | 0.285 | - | - | 1801.3 | - (stored kernel) | 0.285 (MIXED ours whole / arm kernel) | - | 2921.5 | 476.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1751.3 | 1751.3..1751.3 | 1 | 0.293 | - | - | 1751.3 | - (stored kernel) | 0.293 (MIXED ours whole / arm kernel) | - | 2969.2 | 476.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1893.3 | 1893.3..1893.3 | 1 | 0.271 | - | - | 1893.3 | - (stored kernel) | 0.271 (MIXED ours whole / arm kernel) | - | 2650.5 | 320.4 | finite=True, r2=0.981981, rmse=0.155521 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1727.4 | 1727.4..1727.4 | 1 | 0.297 | - | - | 1727.4 | - (stored kernel) | 0.297 (MIXED ours whole / arm kernel) | - | 2698.5 | 320.4 | finite=True, r2=0.981981, rmse=0.155521 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1727.7 ms) = 0.297; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1724.0 ms) = 0.298. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 513.4 | 513.4..513.4 | 1 | - | - | 513.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.747875, rmse=0.544554 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1777.8 | 1777.8..1777.8 | 1 | 0.289 | - | - | 1777.8 | - (stored kernel) | 0.289 (MIXED ours whole / arm kernel) | - | 2921.5 | 476.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1724.0 | 1724.0..1724.0 | 1 | 0.298 | - | - | 1724.0 | - (stored kernel) | 0.298 (MIXED ours whole / arm kernel) | - | 2969.4 | 476.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1881.1 | 1881.1..1881.1 | 1 | 0.273 | - | - | 1881.1 | - (stored kernel) | 0.273 (MIXED ours whole / arm kernel) | - | 2650.5 | 320.4 | finite=True, r2=0.748253, rmse=0.544145 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1727.7 | 1727.7..1727.7 | 1 | 0.297 | - | - | 1727.7 | - (stored kernel) | 0.297 (MIXED ours whole / arm kernel) | - | 2698.4 | 320.4 | finite=True, r2=0.748253, rmse=0.544145 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1710)`, ran on AMD Instinct MI325X (amd, DO) job a1710

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 541.5 | 541.5..541.5 | 1 | - | - | 541.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.802100 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1710 2026-10-11; identity vs nvidia-l40s: MATCH) |
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

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1710)`, ran on AMD Instinct MI325X (amd, DO) job a1710

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 543.0 | 543.0..543.0 | 1 | - | - | 543.0 | - | - (stored whole) | - | - | - | - | recall_at_10=0.994575 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1710 2026-10-11; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | faiss 1.15.1 | opponent | 5684.0 | 5684.0..5684.0 | 1 | 0.096 | - | 5684.0 | 5684.0 | 0.00 (cpu-arm) | 0.096 (whole/whole) | - | 1313.8 | - | recall_at_10=0.993125 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 132.0 | 132.0..132.0 | 1 | - | - | 132.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 2318.5 | 2318.5..2318.5 | 1 | 0.057 | - | 2318.5 | 2318.5 | 0.00 (cpu-arm) | 0.057 (whole/whole) | - | 1470.6 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 13.5 | 13.5..13.5 | 1 | - | - | 13.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 108.6 | 108.6..108.6 | 1 | 0.124 | - | 108.6 | 108.6 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 272.7 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1692)`, ran on AMD Instinct MI325X (amd, DO) job a1692

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 8243.6 | 8243.6..8243.6 | 1 | - | - | 8243.6 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1692 2026-10-11; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 10339.5 | 10339.5..10339.5 | 1 | 0.797 | - | 10339.5 | 10339.5 | 0.00 (cpu-arm) | 0.797 (whole/whole) | - | 2196.3 | - | rel_error_vs_exact=8.63e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1686)`, ran on AMD Instinct MI325X (amd, DO) job a1686

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 22.9 | 22.9..22.9 | 1 | - | - | 22.9 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1686 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 16.2 | 16.2..16.2 | 1 | 1.413 | - | 16.2 | 16.2 | 0.00 (cpu-arm) | 1.413 (whole/whole) | - | 3528.4 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 19.7 | 19.7..19.7 | 1 | - | - | 19.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x16 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 51.9 | 51.9..51.9 | 1 | 0.379 | - | 51.9 | 51.9 | 0.00 (cpu-arm) | 0.379 (whole/whole) | - | 680.8 | - | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 254.5 | 254.5..254.5 | 1 | - | - | 254.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x259 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 139.4 | 139.4..139.4 | 1 | 1.826 | - | 139.4 | 139.4 | 0.00 (cpu-arm) | 1.826 (whole/whole) | - | 6601.1 | - | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1624)`, ran on AMD Instinct MI325X (amd, DO) job a1624

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@15c76449e) | identical | 2.9 | 2.9..2.9 | 1 | - | - | 2.9 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@15c76449e amd/a1624 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 6.8 | 6.8..6.8 | 1 | 0.427 | - | 6.8 | 6.8 | 0.00 (cpu-arm) | 0.427 (whole/whole) | - | 313.1 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1624)`, ran on AMD Instinct MI325X (amd, DO) job a1624

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@15c76449e) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@15c76449e amd/a1624 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 20.6 | 20.6..20.6 | 1 | 0.147 | - | 20.6 | 20.6 | 0.00 (cpu-arm) | 0.147 (whole/whole) | - | 300.5 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lamb / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on AMD Instinct MI325X (amd, DO) job a0873

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 13.4 | 13.4..13.4 | 1 | - | - | 13.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-lamb-synthetic-def/lamb-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 1157.1 | 1157.1..1157.1 | 1 | - | - | 1157.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328699, rmse=0.684414 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 474.5 | 474.5..474.5 | 1 | 2.439 | - | 474.5 | 474.5 | 0.00 (cpu-arm) | 2.439 (whole/whole) | - | 1980.0 | - | finite=True, r2=-1.051e+22, rmse=8.562e+10 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805051 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 32.4 | 32.4..32.4 | 1 | 0.104 | - | 32.4 | 32.4 | 0.00 (cpu-arm) | 0.104 (whole/whole) | - | 303.2 | - | finite=True, r2=0.908983, rmse=4.805057 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 836.6 | 836.6..836.6 | 1 | - | - | 836.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.325506, rmse=0.686040 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 42194.4 | 42194.4..42194.4 | 1 | 0.020 | - | 42194.4 | 42194.4 | 0.00 (cpu-arm) | 0.020 (whole/whole) | - | 8912.8 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1645)`, ran on AMD Instinct MI325X (amd, DO) job a1645

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 198.7 | 198.7..198.7 | 1 | - | - | 198.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909038, rmse=4.803593 | - | main board, one scored run | - | ok (main@2afa942e3 amd/a1645 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 277.2 | 277.2..277.2 | 1 | 0.717 | - | 277.2 | 277.2 | 0.00 (cpu-arm) | 0.717 (whole/whole) | - | 680.5 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 75.3 | 75.3..75.3 | 1 | - | - | 75.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310834, rmse=0.693461 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 461.2 | 461.2..461.2 | 1 | 0.163 | - | 461.2 | 461.2 | 0.00 (cpu-arm) | 0.163 (whole/whole) | - | 1979.4 | - | finite=True, r2=0.310591, rmse=0.693584 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1657)`, ran on AMD Instinct MI325X (amd, DO) job a1657

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908998, rmse=4.804663 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1657 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 32.4 | 32.4..32.4 | 1 | 0.104 | - | 32.4 | 32.4 | 0.00 (cpu-arm) | 0.104 (whole/whole) | - | 303.3 | - | finite=True, r2=0.908998, rmse=4.804669 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1200)`, ran on AMD Instinct MI325X (amd, DO) job a1200

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.9 ms) = 1.389; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.9 ms) = 1.376. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@888f286db) | identical | 1.2 | 1.2..1.2 | 1 | - | - | 1.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1200/work-layernorm-synthetic-def/layernorm-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@888f286db amd/a1200 2026-10-10; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 1.376 | - | - | 0.9 | - (stored kernel) | 1.376 (MIXED ours whole / arm kernel) | - | 1640.7 | 320.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 0.837 | - | - | 1.4 | - (stored kernel) | 0.837 (MIXED ours whole / arm kernel) | - | 2084.3 | 392.1 | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.9 | 0.9..0.9 | 1 | 1.389 | - | - | 0.9 | - (stored kernel) | 1.389 (MIXED ours whole / arm kernel) | - | 1640.3 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 1.022 | - | - | 1.2 | - (stored kernel) | 1.022 (MIXED ours whole / arm kernel) | - | 2011.5 | 392.1 | max_rel_diff_vs_torch_eager_fp32=0.007530, rel_fro_vs_torch_eager_fp32=5.326e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

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

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 936.9 ms) = 0.674; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 863.4 ms) = 0.731. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 631.6 | 631.6..631.6 | 1 | - | - | 631.6 | - | - (stored whole) | - | - | - | - | accuracy=0.967068, logloss=0.079720 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 901.1 | 901.1..901.1 | 1 | 0.701 | - | - | 901.1 | - (stored kernel) | 0.701 (MIXED ours whole / arm kernel) | - | 4838.6 | 502.8 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 863.4 | 863.4..863.4 | 1 | 0.731 | - | - | 863.4 | - (stored kernel) | 0.731 (MIXED ours whole / arm kernel) | - | 4886.7 | 502.8 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 987.5 | 987.5..987.5 | 1 | 0.640 | - | - | 987.5 | - (stored kernel) | 0.640 (MIXED ours whole / arm kernel) | - | 3217.8 | 334.8 | accuracy=0.968913 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 936.9 | 936.9..936.9 | 1 | 0.674 | - | - | 936.9 | - (stored kernel) | 0.674 (MIXED ours whole / arm kernel) | - | 3266.1 | 334.8 | accuracy=0.968913 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 962.8 ms) = 0.655; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 881.9 ms) = 0.715. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 630.7 | 630.7..630.7 | 1 | - | - | 630.7 | - | - (stored whole) | - | - | - | - | accuracy=0.870443, logloss=0.297146 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 887.0 | 887.0..887.0 | 1 | 0.711 | - | - | 887.0 | - (stored kernel) | 0.711 (MIXED ours whole / arm kernel) | - | 5225.7 | 502.8 | accuracy=0.868218 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 881.9 | 881.9..881.9 | 1 | 0.715 | - | - | 881.9 | - (stored kernel) | 0.715 (MIXED ours whole / arm kernel) | - | 4886.4 | 502.8 | accuracy=0.868218 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 976.8 | 976.8..976.8 | 1 | 0.646 | - | - | 976.8 | - (stored kernel) | 0.646 (MIXED ours whole / arm kernel) | - | 3685.3 | 334.8 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 962.8 | 962.8..962.8 | 1 | 0.655 | - | - | 962.8 | - (stored kernel) | 0.655 (MIXED ours whole / arm kernel) | - | 3266.3 | 334.8 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 993.6 ms) = 0.633; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 874.9 ms) = 0.718. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 628.6 | 628.6..628.6 | 1 | - | - | 628.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.979398, rmse=0.166295 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 874.9 | 874.9..874.9 | 1 | 0.718 | - | - | 874.9 | - (stored kernel) | 0.718 (MIXED ours whole / arm kernel) | - | 2920.6 | 502.5 | finite=True, r2=0.981013, rmse=0.159642 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 886.2 | 886.2..886.2 | 1 | 0.709 | - | - | 886.2 | - (stored kernel) | 0.709 (MIXED ours whole / arm kernel) | - | 2968.4 | 502.5 | finite=True, r2=0.981013, rmse=0.159642 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 993.6 | 993.6..993.6 | 1 | 0.633 | - | - | 993.6 | - (stored kernel) | 0.633 (MIXED ours whole / arm kernel) | - | 2649.7 | 334.5 | finite=True, r2=0.981004, rmse=0.159679 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1007.1 | 1007.1..1007.1 | 1 | 0.624 | - | - | 1007.1 | - (stored kernel) | 0.624 (MIXED ours whole / arm kernel) | - | 2697.4 | 334.5 | finite=True, r2=0.981004, rmse=0.159679 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 948.1 ms) = 0.662; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 849.2 ms) = 0.739. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 627.9 | 627.9..627.9 | 1 | - | - | 627.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.754306, rmse=0.537563 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 849.2 | 849.2..849.2 | 1 | 0.739 | - | - | 849.2 | - (stored kernel) | 0.739 (MIXED ours whole / arm kernel) | - | 2930.9 | 502.5 | finite=True, r2=0.751679, rmse=0.540429 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 911.8 | 911.8..911.8 | 1 | 0.689 | - | - | 911.8 | - (stored kernel) | 0.689 (MIXED ours whole / arm kernel) | - | 2965.2 | 502.5 | finite=True, r2=0.751679, rmse=0.540429 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 948.1 | 948.1..948.1 | 1 | 0.662 | - | - | 948.1 | - (stored kernel) | 0.662 (MIXED ours whole / arm kernel) | - | 2649.5 | 334.5 | finite=True, r2=0.751591, rmse=0.540526 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 954.6 | 954.6..954.6 | 1 | 0.658 | - | - | 954.6 | - (stored kernel) | 0.658 (MIXED ours whole / arm kernel) | - | 2697.5 | 334.5 | finite=True, r2=0.751591, rmse=0.540526 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 289.3 | 289.3..289.3 | 1 | - | - | 289.3 | - | - (stored whole) | - | - | - | - | relative_residual=0.849956 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 2526.8 | 2526.8..2526.8 | 1 | 0.114 | - | 2526.8 | 2526.8 | 0.00 (cpu-arm) | 0.114 (whole/whole) | - | 4351.2 | - | relative_residual=0.876106 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3342.1 | 3342.1..3342.1 | 1 | 0.087 | - | - | 3342.1 | - (stored kernel) | 0.087 (MIXED ours whole / arm kernel) | - | 2660.7 | 1819.9 | relative_residual=nan | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 7.9 | 7.9..7.9 | 1 | - | - | 7.9 | - | - (stored whole) | - | - | - | - | relative_residual=0.756366 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 91.5 | 91.5..91.5 | 1 | 0.087 | - | 91.5 | 91.5 | 0.00 (cpu-arm) | 0.087 (whole/whole) | - | 284.0 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 27.5 | 27.5..27.5 | 1 | 0.288 | - | - | 27.5 | - (stored kernel) | 0.288 (MIXED ours whole / arm kernel) | - | 1707.2 | 223.5 | relative_residual=0.756366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1654)`, ran on AMD Instinct MI325X (amd, DO) job a1654

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 312.8 | 312.8..312.8 | 1 | - | - | 312.8 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@c69768593 amd/a1654 2026-10-11; identity vs nvidia-l40s: MATCH) |
| scipy-cpu | scipy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scipy 1.18.1 | opponent | 4553.9 | 4553.9..4553.9 | 1 | 0.069 | - | 4553.9 | 4553.9 | 0.00 (cpu-arm) | 0.069 (whole/whole) | - | 628.0 | - | relative_residual=3.439e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 85.0 | 85.0..85.0 | 1 | 3.678 | - | - | 85.0 | - (stored kernel) | 3.678 (MIXED ours whole / arm kernel) | - | 2080.1 | 710.0 | relative_residual=4.003e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, scipy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1654)`, ran on AMD Instinct MI325X (amd, DO) job a1654

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 313.9 | 313.9..313.9 | 1 | - | - | 313.9 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@c69768593 amd/a1654 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 5413.8 | 5413.8..5413.8 | 1 | 0.058 | - | 5413.8 | 5413.8 | 0.00 (cpu-arm) | 0.058 (whole/whole) | - | 1389.2 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 82.7 | 82.7..82.7 | 1 | 3.795 | - | - | 82.7 | - (stored kernel) | 3.795 (MIXED ours whole / arm kernel) | - | 2104.8 | 712.0 | relative_residual=4.003e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 17.1 | 17.1..17.1 | 1 | - | - | 17.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 123.0 | 123.0..123.0 | 1 | 0.139 | - | 123.0 | 123.0 | 0.00 (cpu-arm) | 0.139 (whole/whole) | - | 2222.0 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 2.3 | 2.3..2.3 | 1 | - | - | 2.3 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 27.7 | 27.7..27.7 | 1 | 0.084 | - | 27.7 | 27.7 | 0.00 (cpu-arm) | 0.084 (whole/whole) | - | 308.5 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1664)`, ran on AMD Instinct MI325X (amd, DO) job a1664

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.8 ms) = 0.481; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.8 ms) = 0.541. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 0.4 | 0.4..0.4 | 1 | - | - | 0.4 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1664 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.541 | - | - | 0.8 | - (stored kernel) | 0.541 (MIXED ours whole / arm kernel) | - | 1609.5 | 288.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.4 | 1.4..1.4 | 1 | 0.284 | - | - | 1.4 | - (stored kernel) | 0.284 (MIXED ours whole / arm kernel) | - | 2138.9 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.8 | 0.8..0.8 | 1 | 0.481 | - | - | 0.8 | - (stored kernel) | 0.481 (MIXED ours whole / arm kernel) | - | 1610.8 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.2 | 1.2..1.2 | 1 | 0.340 | - | - | 1.2 | - (stored kernel) | 0.340 (MIXED ours whole / arm kernel) | - | 1976.3 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1212)`, ran on AMD Instinct MI325X (amd, DO) job a1212

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.0 ms) = 0.673; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 0.869. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1212 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 0.7 | 0.7..0.7 | 1 | 0.869 | - | - | 0.7 | - (stored kernel) | 0.869 (MIXED ours whole / arm kernel) | - | 1801.1 | 639.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 0.672 | - | - | 1.0 | - (stored kernel) | 0.672 (MIXED ours whole / arm kernel) | - | 2484.0 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.0 | 1.0..1.0 | 1 | 0.673 | - | - | 1.0 | - (stored kernel) | 0.673 (MIXED ours whole / arm kernel) | - | 1801.3 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 1.3 | 1.3..1.3 | 1 | 0.509 | - | - | 1.3 | - (stored kernel) | 0.509 (MIXED ours whole / arm kernel) | - | 2127.4 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1688)`, ran on AMD Instinct MI325X (amd, DO) job a1688

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 344.1 | 344.1..344.1 | 1 | - | - | 344.1 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.116696 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1688 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 200.4 | 200.4..200.4 | 1 | 1.717 | - | 200.4 | 200.4 | 0.00 (cpu-arm) | 1.717 (whole/whole) | - | 1178.2 | - | ari_vs_ours=0.622446, n_clusters=8, silhouette=0.111849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1688)`, ran on AMD Instinct MI325X (amd, DO) job a1688

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 48.2 | 48.2..48.2 | 1 | - | - | 48.2 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.138060 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1688 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 26.5 | 26.5..26.5 | 1 | 1.815 | - | 26.5 | 26.5 | 0.00 (cpu-arm) | 1.815 (whole/whole) | - | 280.6 | - | ari_vs_ours=0.525151, n_clusters=8, silhouette=0.165473 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### moe / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1682)`, ran on AMD Instinct MI325X (amd, DO) job a1682

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.7 ms) = 5.608; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 4.5 ms) = 3.412. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 15.2 | 15.2..15.2 | 1 | - | - | 15.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/a1682/work-moe-synthetic-def/moe-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1682 2026-10-11; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.5 | 4.5..4.5 | 1 | 3.412 | - | - | 4.5 | - (stored kernel) | 3.412 (MIXED ours whole / arm kernel) | - | 3404.0 | 537.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 5.0 | 5.0..5.0 | 1 | 3.054 | - | - | 5.0 | - (stored kernel) | 3.054 (MIXED ours whole / arm kernel) | - | 3676.5 | 514.6 | max_rel_diff_vs_torch_eager_fp32=0.017192, rel_fro_vs_torch_eager_fp32=1.088e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.7 | 2.7..2.7 | 1 | 5.608 | - | - | 2.7 | - (stored kernel) | 5.608 (MIXED ours whole / arm kernel) | - | 2792.9 | 641.3 | max_rel_diff_vs_torch_eager_fp32=22778.779951, rel_fro_vs_torch_eager_fp32=0.055222 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 3.1 | 3.1..3.1 | 1 | 4.935 | - | - | 3.1 | - (stored kernel) | 4.935 (MIXED ours whole / arm kernel) | - | 3061.3 | 501.4 | max_rel_diff_vs_torch_eager_fp32=22796.112469, rel_fro_vs_torch_eager_fp32=0.055199 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 67.5 | 67.5..67.5 | 1 | - | - | 67.5 | - | - (stored whole) | - | - | - | - | accuracy=0.853620, logloss=3.628565 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 209.1 | 209.1..209.1 | 1 | 0.323 | - | 209.1 | 209.1 | 0.00 (cpu-arm) | 0.323 (whole/whole) | - | 3958.7 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 27.9 | 27.9..27.9 | 1 | - | - | 27.9 | - | - (stored whole) | - | - | - | - | accuracy=0.723160, logloss=0.590725 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 38.6 | 38.6..38.6 | 1 | 0.723 | - | 38.6 | 38.6 | 0.00 (cpu-arm) | 0.723 (whole/whole) | - | 450.2 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 7.0 ms) = 0.572. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 4.0 | 4.0..4.0 | 1 | - | - | 4.0 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 7.0 | 7.0..7.0 | 1 | 0.572 | - | - | 7.0 | - (stored kernel) | 0.572 (MIXED ours whole / arm kernel) | - | 2624.0 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 91.7 | 91.7..91.7 | 1 | 0.044 | - | - | 91.7 | - (stored kernel) | 0.044 (MIXED ours whole / arm kernel) | - | 3136.2 | 896.0 | rel_fro_vs_torch_eager_fp32=1.464e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nearest-centroid / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1684)`, ran on AMD Instinct MI325X (amd, DO) job a1684

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 38.1 | 38.1..38.1 | 1 | - | - | 38.1 | - | - (stored whole) | - | - | - | - | accuracy=0.852610, logloss=4.299485 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1684 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 669.1 | 669.1..669.1 | 1 | 0.057 | - | 669.1 | 669.1 | 0.00 (cpu-arm) | 0.057 (whole/whole) | - | 4526.8 | - | accuracy=0.852610, logloss=4.117692 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nearest-centroid / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1684)`, ran on AMD Instinct MI325X (amd, DO) job a1684

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 21.4 | 21.4..21.4 | 1 | - | - | 21.4 | - | - (stored whole) | - | - | - | - | accuracy=0.666700, logloss=0.781905 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1684 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 116.3 | 116.3..116.3 | 1 | 0.184 | - | 116.3 | 116.3 | 0.00 (cpu-arm) | 0.184 (whole/whole) | - | 460.1 | - | accuracy=0.666750, logloss=0.781690 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 21.1 | 21.1..21.1 | 1 | - | - | 21.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x119 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 13.3 | 13.3..13.3 | 1 | 1.595 | - | 13.3 | 13.3 | 0.00 (cpu-arm) | 1.595 (whole/whole) | - | 539.3 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 17.7 | 17.7..17.7 | 1 | - | - | 17.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x508 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 8.0 | 8.0..8.0 | 1 | 2.207 | - | 8.0 | 8.0 | 0.00 (cpu-arm) | 2.207 (whole/whole) | - | 1407.9 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 407.2 | 407.2..407.2 | 1 | - | - | 407.2 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.438855 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 303.9 | 303.9..303.9 | 1 | 1.340 | - | 303.9 | 303.9 | 0.00 (cpu-arm) | 1.340 (whole/whole) | - | 283.6 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 735.3 | 735.3..735.3 | 1 | - | - | 735.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.150860 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 770.4 | 770.4..770.4 | 1 | 0.954 | - | 770.4 | 770.4 | 0.00 (cpu-arm) | 0.954 (whole/whole) | - | 283.5 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 21.8 | 21.8..21.8 | 1 | - | - | 21.8 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 13.6 | 13.6..13.6 | 1 | 1.606 | - | 13.6 | 13.6 | 0.00 (cpu-arm) | 1.606 (whole/whole) | - | 267.7 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1706)`, ran on AMD Instinct MI325X (amd, DO) job a1706

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 17.5 | 17.5..17.5 | 1 | - | - | 17.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1706 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 7.4 | 7.4..7.4 | 1 | 2.347 | - | 7.4 | 7.4 | 0.00 (cpu-arm) | 2.347 (whole/whole) | - | 246.0 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1690)`, ran on AMD Instinct MI325X (amd, DO) job a1690

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1975.6 | 1975.6..1975.6 | 1 | - | - | 1975.6 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1690 2026-10-11; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 22453.9 | 22453.9..22453.9 | 1 | 0.088 | - | 22453.9 | 22453.9 | 0.00 (cpu-arm) | 0.088 (whole/whole) | - | 1842.9 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1690)`, ran on AMD Instinct MI325X (amd, DO) job a1690

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 34.4 | 34.4..34.4 | 1 | - | - | 34.4 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1690 2026-10-11; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | shap 0.51.0 | opponent | 104.1 | 104.1..104.1 | 1 | 0.330 | - | 104.1 | 104.1 | 0.00 (cpu-arm) | 0.330 (whole/whole) | - | 478.9 | - | rel_error_vs_exact=1.218e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 683.4 | 683.4..683.4 | 1 | - | - | 683.4 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.485e-07 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 4646.6 | 4646.6..4646.6 | 1 | 0.147 | - | 4646.6 | 4646.6 | 0.00 (cpu-arm) | 0.147 (whole/whole) | - | 8532.1 | - | relative_gram_difference=2.462e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2664.9 | 2664.9..2664.9 | 1 | 0.256 | - | - | 2664.9 | - (stored kernel) | 0.256 (MIXED ours whole / arm kernel) | - | 2740.9 | 2520.9 | relative_gram_difference=0.0001695 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 16.0 | 16.0..16.0 | 1 | - | - | 16.0 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.714e-07 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 177.3 | 177.3..177.3 | 1 | 0.090 | - | 177.3 | 177.3 | 0.00 (cpu-arm) | 0.090 (whole/whole) | - | 478.4 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 27.0 | 27.0..27.0 | 1 | 0.592 | - | - | 27.0 | - (stored kernel) | 0.592 (MIXED ours whole / arm kernel) | - | 1594.3 | 126.0 | relative_gram_difference=9.54e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 125.2 | 125.2..125.2 | 1 | - | - | 125.2 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.0002359 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 812.6 | 812.6..812.6 | 1 | 0.154 | - | 812.6 | 812.6 | 0.00 (cpu-arm) | 0.154 (whole/whole) | - | 1710.8 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 206.8 | 206.8..206.8 | 1 | 0.605 | - | - | 206.8 | - (stored kernel) | 0.605 (MIXED ours whole / arm kernel) | - | 4516.2 | 1018.1 | relative_reconstruction_error=0.0002359 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 46.1 | 46.1..46.1 | 1 | - | - | 46.1 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.027197 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 375.4 | 375.4..375.4 | 1 | 0.123 | - | 375.4 | 375.4 | 0.00 (cpu-arm) | 0.123 (whole/whole) | - | 476.9 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 120.8 | 120.8..120.8 | 1 | 0.382 | - | - | 120.8 | - (stored kernel) | 0.382 (MIXED ours whole / arm kernel) | - | 3715.1 | 228.0 | relative_reconstruction_error=0.027197 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1721)`, ran on AMD Instinct MI325X (amd, DO) job a1721

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.0 ms) = 7.553; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.0 ms) = 7.666. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@7924718fc) | identical | 15.5 | 15.5..15.5 | 1 | - | - | 15.5 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (mojolearn: MOJOLEARN_VENDOR=cpu; no GPU set is loaded, on request (flat layout)); see /root/lq/out/a1721/work-resnet-block-synthetic-def/resnet-block-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@7924718fc amd/a1721 2026-10-11; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.0 | 2.0..2.0 | 1 | 7.666 | - | - | 2.0 | - (stored kernel) | 7.666 (MIXED ours whole / arm kernel) | - | 2322.0 | 547.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.1 | 2.1..2.1 | 1 | 7.332 | - | - | 2.1 | - (stored kernel) | 7.332 (MIXED ours whole / arm kernel) | - | 2382.8 | 510.1 | max_rel_diff_vs_torch_eager_fp32=0.357628, rel_fro_vs_torch_eager_fp32=1.809e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.0 | 2.0..2.0 | 1 | 7.553 | - | - | 2.0 | - (stored kernel) | 7.553 (MIXED ours whole / arm kernel) | - | 2472.7 | 471.0 | max_rel_diff_vs_torch_eager_fp32=24577.140808, rel_fro_vs_torch_eager_fp32=0.004174 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2.1 | 2.1..2.1 | 1 | 7.390 | - | - | 2.1 | - (stored kernel) | 7.390 (MIXED ours whole / arm kernel) | - | 2368.7 | 434.5 | max_rel_diff_vs_torch_eager_fp32=18499.135971, rel_fro_vs_torch_eager_fp32=0.003425 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1661)`, ran on AMD Instinct MI325X (amd, DO) job a1661

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 62879.7 | 62879.7..62879.7 | 1 | - | - | 62879.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328684, rmse=0.684422 | - | main board, one scored run | - | ok (main@6cd6bed7f amd/a1661 2026-10-11; identity vs the nvidia columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | scikit-learn 1.7.2 | opponent | 78354.8 | 78354.8..78354.8 | 1 | 0.802 | - | 78354.8 | 78354.8 | 0.00 (cpu-arm) | 0.802 (whole/whole) | - | 8705.1 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 4.9 ms) = 0.692. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@0a7b206f1 amd/a1156 2026-10-09; identity vs the nvidia columns: n/a) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 4.9 | 4.9..4.9 | 1 | 0.692 | - | - | 4.9 | - (stored kernel) | 0.692 (MIXED ours whole / arm kernel) | - | 2619.7 | 896.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 53.8 | 53.8..53.8 | 1 | 0.063 | - | - | 53.8 | - (stored kernel) | 0.063 (MIXED ours whole / arm kernel) | - | 3015.5 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 804.1 ms) = 0.380; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 797.3 ms) = 0.384. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 305.9 | 305.9..305.9 | 1 | - | - | 305.9 | - | - (stored whole) | - | - | - | - | accuracy=0.967828, logloss=0.079029 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 837.8 | 837.8..837.8 | 1 | 0.365 | - | - | 837.8 | - (stored kernel) | 0.365 (MIXED ours whole / arm kernel) | - | 4832.5 | 260.6 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 797.3 | 797.3..797.3 | 1 | 0.384 | - | - | 797.3 | - (stored kernel) | 0.384 (MIXED ours whole / arm kernel) | - | 4881.4 | 260.6 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 936.2 | 936.2..936.2 | 1 | 0.327 | - | - | 936.2 | - (stored kernel) | 0.327 (MIXED ours whole / arm kernel) | - | 3218.4 | 212.6 | accuracy=0.953396 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 804.1 | 804.1..804.1 | 1 | 0.380 | - | - | 804.1 | - (stored kernel) | 0.380 (MIXED ours whole / arm kernel) | - | 3267.1 | 212.6 | accuracy=0.953396 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 932.6 ms) = 0.329; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 829.9 ms) = 0.369. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 306.6 | 306.6..306.6 | 1 | - | - | 306.6 | - | - (stored whole) | - | - | - | - | accuracy=0.862684, logloss=0.313008 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 849.1 | 849.1..849.1 | 1 | 0.361 | - | - | 849.1 | - (stored kernel) | 0.361 (MIXED ours whole / arm kernel) | - | 5176.5 | 260.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 829.9 | 829.9..829.9 | 1 | 0.369 | - | - | 829.9 | - (stored kernel) | 0.369 (MIXED ours whole / arm kernel) | - | 4881.0 | 260.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 932.6 | 932.6..932.6 | 1 | 0.329 | - | - | 932.6 | - (stored kernel) | 0.329 (MIXED ours whole / arm kernel) | - | 3568.2 | 212.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 952.4 | 952.4..952.4 | 1 | 0.322 | - | - | 952.4 | - (stored kernel) | 0.322 (MIXED ours whole / arm kernel) | - | 3267.2 | 212.6 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 939.1 ms) = 0.320; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 856.8 ms) = 0.350. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 300.1 | 300.1..300.1 | 1 | - | - | 300.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.978602, rmse=0.169476 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 856.8 | 856.8..856.8 | 1 | 0.350 | - | - | 856.8 | - (stored kernel) | 0.350 (MIXED ours whole / arm kernel) | - | 2934.7 | 260.3 | finite=True, r2=0.977348, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 867.3 | 867.3..867.3 | 1 | 0.346 | - | - | 867.3 | - (stored kernel) | 0.346 (MIXED ours whole / arm kernel) | - | 2967.2 | 260.3 | finite=True, r2=0.977348, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 939.1 | 939.1..939.1 | 1 | 0.320 | - | - | 939.1 | - (stored kernel) | 0.320 (MIXED ours whole / arm kernel) | - | 2642.9 | 212.3 | finite=True, r2=0.977383, rmse=0.174238 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 952.3 | 952.3..952.3 | 1 | 0.315 | - | - | 952.3 | - (stored kernel) | 0.315 (MIXED ours whole / arm kernel) | - | 2691.2 | 212.3 | finite=True, r2=0.977383, rmse=0.174238 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1211)`, ran on AMD Instinct MI325X (amd, DO) job a1211

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 916.4 ms) = 0.328; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 845.1 ms) = 0.356. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 300.7 | 300.7..300.7 | 1 | - | - | 300.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.743904, rmse=0.548825 | - | main board, one scored run | - | ok (main@1665b5626 amd/a1211 2026-10-10; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 845.1 | 845.1..845.1 | 1 | 0.356 | - | - | 845.1 | - (stored kernel) | 0.356 (MIXED ours whole / arm kernel) | - | 2933.1 | 260.3 | finite=True, r2=0.738796, rmse=0.554271 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 857.8 | 857.8..857.8 | 1 | 0.351 | - | - | 857.8 | - (stored kernel) | 0.351 (MIXED ours whole / arm kernel) | - | 2966.9 | 260.3 | finite=True, r2=0.738796, rmse=0.554271 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 916.4 | 916.4..916.4 | 1 | 0.328 | - | - | 916.4 | - (stored kernel) | 0.328 (MIXED ours whole / arm kernel) | - | 2643.1 | 212.3 | finite=True, r2=0.738886, rmse=0.554176 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 973.8 | 973.8..973.8 | 1 | 0.309 | - | - | 973.8 | - (stored kernel) | 0.309 (MIXED ours whole / arm kernel) | - | 2691.9 | 212.3 | finite=True, r2=0.738886, rmse=0.554176 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 737.2 | 737.2..737.2 | 1 | - | - | 737.2 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=20.498972, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 3741.4 | 3741.4..3741.4 | 1 | 0.197 | - | 3741.4 | 3741.4 | 0.00 (cpu-arm) | 0.197 (whole/whole) | - | 8708.9 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 2939.0 | 2939.0..2939.0 | 1 | 0.251 | - | - | 2939.0 | - (stored kernel) | 0.251 (MIXED ours whole / arm kernel) | - | 2916.8 | 3360.7 | max_rel_singular_value_error=2.707e+08, relative_reconstruction_error_100k_rows=0.026534 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1670)`, ran on AMD Instinct MI325X (amd, DO) job a1670

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 19.1 | 19.1..19.1 | 1 | - | - | 19.1 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1670 2026-10-11; identity vs nvidia-l40s: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | numpy 2.5.3 | opponent | 130.5 | 130.5..130.5 | 1 | 0.146 | - | 130.5 | 130.5 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 485.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | torch 2.13.0+rocm7.1 | opponent | 272.1 | 272.1..272.1 | 1 | 0.070 | - | - | 272.1 | - (stored kernel) | 0.070 (MIXED ours whole / arm kernel) | - | 1865.6 | 168.0 | max_rel_singular_value_error=4.409e-05, relative_reconstruction_error_100k_rows=0.003043 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1655)`, ran on AMD Instinct MI325X (amd, DO) job a1655

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 620.3 | 620.3..620.3 | 1 | - | - | 620.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@c69768593 amd/a1655 2026-10-11; identity vs nvidia-l40s: MATCH) |
| gpytorch-gpu | gpytorch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | gpytorch 1.15.2 | opponent | 16.2 | 16.2..16.2 | 1 | 38.180 | - | - | 16.2 | - (stored kernel) | 38.180 (MIXED ours whole / arm kernel) | - | 5943.4 | 1687.8 | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | gpytorch 1.15.2 | opponent | 672.0 | 672.0..672.0 | 1 | 0.923 | - | 672.0 | 672.0 | 0.00 (cpu-arm) | 0.923 (whole/whole) | - | 3457.1 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1655)`, ran on AMD Instinct MI325X (amd, DO) job a1655

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 715.9 | 715.9..715.9 | 1 | - | - | 715.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.194982, rmse=17.723482 | - | main board, one scored run | - | ok (main@c69768593 amd/a1655 2026-10-11; identity vs nvidia-l40s: MATCH) |
| gpytorch-gpu | gpytorch | gpu | AMD Instinct Mi325X VF (host mojolearn-steward-do-amd) | gpytorch 1.15.2 | opponent | 15.4 | 15.4..15.4 | 1 | 46.468 | - | - | 15.4 | - (stored kernel) | 46.468 (MIXED ours whole / arm kernel) | - | 4970.7 | 1590.9 | finite=True, r2=-0.209526, rmse=17.831014 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | gpytorch 1.15.2 | opponent | 504.6 | 504.6..504.6 | 1 | 1.419 | - | 504.6 | 504.6 | 0.00 (cpu-arm) | 1.419 (whole/whole) | - | 2461.6 | - | finite=True, r2=-0.209532, rmse=17.831057 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 122.3 | 122.3..122.3 | 1 | - | - | 122.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436610 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 104.8 | 104.8..104.8 | 1 | 1.167 | - | 104.8 | 104.8 | 0.00 (cpu-arm) | 1.167 (whole/whole) | - | 283.6 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 38.6 | 38.6..38.6 | 1 | 3.169 | - | 38.6 | 38.6 | 0.00 (cpu-arm) | 3.169 (whole/whole) | - | 57.6 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1679)`, ran on AMD Instinct MI325X (amd, DO) job a1679

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 390.7 | 390.7..390.7 | 1 | - | - | 390.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.020604 | - | main board, one scored run | - | ok (main@a1c96a5ba amd/a1679 2026-10-11; identity vs the nvidia columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsforecast 2.1.1 | opponent | 548.4 | 548.4..548.4 | 1 | 0.712 | - | 548.4 | 548.4 | 0.00 (cpu-arm) | 0.712 (whole/whole) | - | 283.4 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9575F 64-Core Processor (host mojolearn-steward-do-amd, AMD Instinct MI325X box) | statsmodels 0.15.0 | opponent | 52.3 | 52.3..52.3 | 1 | 7.471 | - | 52.3 | 52.3 | 0.00 (cpu-arm) | 7.471 (whole/whole) | - | 58.1 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1710)`, ran on AMD Instinct MI325X (amd, DO) job a1710

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1216.2 | 1216.2..1216.2 | 1 | - | - | 1216.2 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.992124 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1710 2026-10-11; identity vs nvidia-l40s: MATCH) |
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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/amd-results.txt (amd/a1711)`, ran on AMD Instinct MI325X (amd, DO) job a1711

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | AMD Instinct MI325X (amd, DO) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1047.3 | 1047.3..1047.3 | 1 | - | - | 1047.3 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.998921 | - | main board, one scored run | - | ok (main@c8ef6c66d amd/a1711 2026-10-11; identity vs nvidia-l40s: MATCH) |
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

