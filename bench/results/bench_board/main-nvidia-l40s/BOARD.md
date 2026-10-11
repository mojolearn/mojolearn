# mojolearn benchmark board

Generated 2026-10-11T21:57:50Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD nvidia-l40s, version label main@7924718fc. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 211 races; oldest cell main@89485bc96 (2026-10-08T13:27:06Z), newest cell main@7924718fc (2026-10-11T21:48:42Z). Boxes: NVIDIA L40S (nv, RunPod), NVIDIA L40S (nv2, RunPod).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 149 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: one scored run per cell (lq RACE ALGOS lines, lq CMD bench_board summaries); the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 112, n/a 99).

> Opponents: copied from the stored opponent boards (release-board-resume-r2), never re-run here; `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 4 races.

> Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax. A cell of ours whose output hash equals a copied opponent's shows quality identical_to=<arm> (the same bits) where the own-host reference gave none.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 112, n/a 99.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 0 failed runs.


## Box

| field | value |
|---|---|
| vendor / API | nvidia / cuda |
| GPU | NVIDIA L40S |
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

Races: 211 planned, 211 done, 0 failed, 0 unsupported, 0 pending. Cells: 778 (REFUSED 34, ok 744).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adamax | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.395e-09 |
| algos | ard | istella | r2 (higher is better) | - | -0.124856 | sklearn-cpu 0.327402 |
| algos | ard | istella | rmse (lower is better) | - | 0.885949 | sklearn-cpu 0.685075 |
| algos | autoarima | synthetic | forecast_rmse (lower is better) | - | 3.260829 | statsforecast-cpu 17.543262; cuml-gpu - |
| algos | autoarima | taxi-hourly | forecast_rmse (lower is better) | - | 73.388358 | statsforecast-cpu 68.213954; cuml-gpu 106.817545 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002792; torch-eager-tf32 0.000000; torch-compile-tf32 0.002792; torch-eager-bf16 0.000000; torch-compile-bf16 0.002792 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.109e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.109e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.109e-08 |
| algos | batchnorm2d | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001863; torch-eager-tf32 0.000000; torch-compile-tf32 0.001863; torch-eager-bf16 0.000000; torch-compile-bf16 0.001863 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 4.327e-08; torch-eager-tf32 0.000000; torch-compile-tf32 4.327e-08; torch-eager-bf16 0.000000; torch-compile-bf16 4.327e-08 |
| algos | bayesian-ridge | istella | r2 (higher is better) | - | 0.317909 | sklearn-cpu -395441.818968 |
| algos | bayesian-ridge | istella | rmse (lower is better) | - | 0.689893 | sklearn-cpu 525.293799 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | - | 0.794050 | sklearn-cpu 0.794050; cuml-gpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | - | 5.350625 | sklearn-cpu 4.278741; cuml-gpu 5.350631 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | 0.755560 | sklearn-cpu 0.755560; cuml-gpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | 0.557803 | sklearn-cpu 0.557802; cuml-gpu 0.557803 |
| algos | binarizer | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | binarizer | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | binarizer | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | binarizer | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | categorical-nb | istella | accuracy (higher is better) | - | 0.838850 | sklearn-cpu 0.838850; cuml-gpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | - | 0.412625 | sklearn-cpu 0.412625; cuml-gpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | 0.765850 | sklearn-cpu 0.765850; cuml-gpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | 0.538866 | sklearn-cpu 0.538866; cuml-gpu 0.538866 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | numpy-cpu 3.928e-08; torch-gpu 1.509e-07; cupy-gpu 1.365e-07 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | - | 1.000000 | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000; torch-eager-tf32 1.000000; torch-compile-tf32 1.000000; torch-eager-bf16 1.000000; torch-compile-bf16 1.000000 |
| algos | complement-nb | istella | accuracy (higher is better) | - | 0.849360 | sklearn-cpu 0.849350; cuml-gpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | - | 3.762524 | sklearn-cpu 3.174763; cuml-gpu 3.763060 |
| algos | complement-nb | taxi | accuracy (higher is better) | - | 0.678020 | sklearn-cpu 0.678030; cuml-gpu 0.678060 |
| algos | complement-nb | taxi | logloss (lower is better) | - | 0.715492 | sklearn-cpu 0.715493; cuml-gpu 0.715531 |
| algos | complement-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067; cuml-gpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | 0.559491 | sklearn-cpu 0.557285; cuml-gpu 0.559490 |
| algos | connected-components | istella | n_components | - | 81 | networkx-cpu 81; cugraph-gpu 81 |
| algos | connected-components | istella | ari_vs_networkx | - | - | networkx-cpu -; cugraph-gpu 1.000000 |
| algos | connected-components | taxi | n_components | - | 588 | networkx-cpu 588; cugraph-gpu 588 |
| algos | connected-components | taxi | ari_vs_networkx | - | - | networkx-cpu -; cugraph-gpu 1.000000 |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 3417.849541; torch-compile-bf16 3524.661064 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.003296; torch-compile-bf16 0.003294 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.506768; torch-eager-tf32 382.931903; torch-compile-tf32 382.931903; torch-eager-bf16 3418.337554; torch-compile-bf16 3433.596343 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.007e-07; torch-eager-tf32 0.0003021; torch-compile-tf32 0.0003021; torch-eager-bf16 0.003382; torch-compile-bf16 0.003380 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | - | 13.931153 | statsmodels-cpu 26.588738; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | - | 96.690449 | statsmodels-cpu 196.955273; statsforecast-cpu 96.685568 |
| algos | dynamic-optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.436278 | statsforecast-cpu 1.436045 |
| algos | dynamic-optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.086628 | statsforecast-cpu 49.314797 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 3.692e-07 | numpy-cpu 3.49e-08; torch-gpu 1.044e-06; cupy-gpu 1.044e-06 |
| algos | eigh | synthetic | relative_residual | - | 6.672e-07 | numpy-cpu 2.824e-08; torch-gpu 1.016e-06; cupy-gpu 1.016e-06 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | enet-cv | istella | r2 (higher is better) | - | 0.326805 | sklearn-cpu 0.326805 |
| algos | enet-cv | istella | rmse (lower is better) | - | 0.685379 | sklearn-cpu 0.685379 |
| algos | enet-cv | taxi | r2 (higher is better) | - | 0.909004 | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | - | 4.804486 | sklearn-cpu 4.804486 |
| algos | gaussian-nb | istella | accuracy (higher is better) | - | 0.876570 | sklearn-cpu 0.876530; cuml-gpu 0.876570 |
| algos | gaussian-nb | istella | logloss (lower is better) | - | 3.574420 | sklearn-cpu 3.417392; cuml-gpu 3.416741 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | 0.719820 | sklearn-cpu 0.719900; cuml-gpu 0.719810 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | 1.132249 | sklearn-cpu 1.133898; cuml-gpu 1.132317 |
| algos | gaussian-rp | istella | mean_abs_distortion | - | 0.680693 | sklearn-cpu 0.177966; cuml-gpu 0.443920 |
| algos | gaussian-rp | taxi | mean_abs_distortion | - | 0.345752 | sklearn-cpu 0.339791; cuml-gpu 0.302259 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.016245; torch-eager-tf32 281.178644; torch-compile-tf32 281.174567; torch-eager-bf16 2718.608563; torch-compile-bf16 2718.611825 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.041e-07; torch-eager-tf32 0.0002661; torch-compile-tf32 0.0002661; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002328; torch-eager-tf32 0.001397; torch-compile-tf32 0.002215; torch-eager-bf16 1509.509282; torch-compile-bf16 1509.509049 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.806e-08; torch-eager-tf32 7.217e-08; torch-compile-tf32 9.792e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.009731; torch-eager-tf32 0.000000; torch-compile-tf32 0.009731; torch-eager-bf16 0.000000; torch-compile-bf16 0.009731 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.64e-08; torch-eager-tf32 0.000000; torch-compile-tf32 9.64e-08; torch-eager-bf16 0.000000; torch-compile-bf16 9.64e-08 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.119209; torch-eager-tf32 430.904329; torch-compile-tf32 430.904329; torch-eager-bf16 3907.084465; torch-compile-bf16 3678.202629 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.699e-08; torch-eager-tf32 0.0002969; torch-compile-tf32 0.0002969; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.238419; torch-eager-tf32 0.083333; torch-compile-tf32 1475.334167; torch-eager-bf16 5460.333333; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.978e-08; torch-eager-tf32 4.164e-08; torch-compile-tf32 0.0002145; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | 0.971625 | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-tf32 0.971842; torch-compile-tf32 0.971842; torch-eager-bf16 0.971951; torch-compile-bf16 0.971951 |
| algos | gru-clf | synthetic | logloss (lower is better) | - | 0.070054 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | 0.865723 | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-tf32 0.865668; torch-compile-tf32 0.865668; torch-eager-bf16 0.865668; torch-compile-bf16 0.865668 |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | - | 0.303537 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | - | 0.982337 | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-tf32 0.981946; torch-compile-tf32 0.981946; torch-eager-bf16 0.981898; torch-compile-bf16 0.981898 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | 0.153975 | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-tf32 0.155672; torch-compile-tf32 0.155672; torch-eager-bf16 0.155878; torch-compile-bf16 0.155878 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | 0.747875 | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-tf32 0.748219; torch-compile-tf32 0.748219; torch-eager-bf16 0.748325; torch-compile-bf16 0.748325 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | 0.544554 | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-tf32 0.544182; torch-compile-tf32 0.544182; torch-eager-bf16 0.544067; torch-compile-bf16 0.544067 |
| algos | incremental-pca | istella | explained_variance_fraction | - | 1.000000 | sklearn-cpu 1.000000; cuml-gpu 1.000000 |
| algos | incremental-pca | taxi | explained_variance_fraction | - | 0.999995 | sklearn-cpu 0.999995; cuml-gpu 0.999995 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | 0.802100 | faiss-cpu -; cuvs-gpu 0.798425 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.981250 | faiss-cpu 0.979450; cuvs-gpu 0.977950 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | - | 0.994575 | faiss-cpu -; cuvs-gpu 0.995050 |
| algos | ivf-refine | taxi | recall_at_10 (higher is better) | - | 0.999675 | faiss-cpu 0.999375; cuvs-gpu 0.998975 |
| algos | ivf-sq | taxi | recall_at_10 (higher is better) | - | 0.934975 | faiss-cpu 0.857050; cuvs-gpu 0.779425 |
| algos | kbins | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 1.000000 |
| algos | kbins | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.0006747 |
| algos | kbins | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | kbins | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | 8.307e-08 | shap-cpu 8.574e-15; cuml-gpu 0.030429 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | 1.056e-07 | shap-cpu 8.154e-15; cuml-gpu 7.647e-07 |
| algos | knn-imputer | istella | masked_rmse | - | 323953.237332 | sklearn-cpu 986208.700423 |
| algos | knn-imputer | taxi | masked_rmse | - | 6.151696 | sklearn-cpu 5.263919 |
| algos | label-binarizer | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | label-binarizer | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | label-binarizer | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | label-binarizer | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | label-encoder | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | label-encoder | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | label-encoder | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | label-encoder | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | lars | istella | r2 (higher is better) | - | 0.328699 | sklearn-cpu -4.245e+13; cuml-gpu 0.328088 |
| algos | lars | istella | rmse (lower is better) | - | 0.684414 | sklearn-cpu 5.442e+06; cuml-gpu 0.684726 |
| algos | lars | taxi | r2 (higher is better) | - | 0.908983 | sklearn-cpu 0.908983; cuml-gpu 0.908983 |
| algos | lars | taxi | rmse (lower is better) | - | 4.805051 | sklearn-cpu 4.805055; cuml-gpu 4.805052 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.325506 | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.686040 | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909038 | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803593 | sklearn-cpu 4.803593 |
| algos | lasso-lars | istella | r2 (higher is better) | - | 0.310834 | sklearn-cpu 0.310816 |
| algos | lasso-lars | istella | rmse (lower is better) | - | 0.693461 | sklearn-cpu 0.693470 |
| algos | lasso-lars | taxi | r2 (higher is better) | - | 0.908998 | sklearn-cpu 0.908998 |
| algos | lasso-lars | taxi | rmse (lower is better) | - | 4.804663 | sklearn-cpu 4.804667 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.006419; torch-eager-tf32 0.000000; torch-compile-tf32 0.006419; torch-eager-bf16 0.000000; torch-compile-bf16 0.006419 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.316e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.316e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.316e-08 |
| algos | lda-clf | istella | accuracy (higher is better) | - | 0.912830 | sklearn-cpu 0.899510 |
| algos | lda-clf | istella | logloss (lower is better) | - | 0.235875 | sklearn-cpu 0.438978 |
| algos | lda-clf | taxi | accuracy (higher is better) | - | 0.762530 | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | - | 0.539763 | sklearn-cpu 0.539767 |
| algos | louvain | istella | modularity | - | 0.911187 | networkx-cpu 0.908460; cugraph-gpu 0.909430 |
| algos | louvain | istella | n_communities | - | 40 | networkx-cpu 40; cugraph-gpu 41 |
| algos | louvain | taxi | modularity | - | 0.941953 | networkx-cpu 0.940781; cugraph-gpu 0.941795 |
| algos | louvain | taxi | n_communities | - | 58 | networkx-cpu 56; cugraph-gpu 62 |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | 0.967068 | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-tf32 0.968696; torch-compile-tf32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-clf | synthetic | logloss (lower is better) | - | 0.079720 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | 0.870443 | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-tf32 0.868164; torch-compile-tf32 0.868164; torch-eager-bf16 0.868327; torch-compile-bf16 0.868327 |
| algos | lstm-clf | taxi-hourly | logloss (lower is better) | - | 0.297146 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | 0.979398 | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-tf32 0.981013; torch-compile-tf32 0.981013; torch-eager-bf16 0.980998; torch-compile-bf16 0.980998 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | 0.166295 | torch-eager-fp32 0.159641; torch-compile-fp32 0.159641; torch-eager-tf32 0.159642; torch-compile-tf32 0.159642; torch-eager-bf16 0.159706; torch-compile-bf16 0.159706 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | 0.754306 | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-tf32 0.751680; torch-compile-tf32 0.751680; torch-eager-bf16 0.751712; torch-compile-bf16 0.751712 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | 0.537563 | torch-eager-fp32 0.540430; torch-compile-fp32 0.540430; torch-eager-tf32 0.540429; torch-compile-tf32 0.540429; torch-eager-bf16 0.540394; torch-compile-bf16 0.540394 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | numpy-cpu 0.876581; torch-gpu nan; cupy-gpu 0.849957 |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | numpy-cpu 0.756366; torch-gpu 0.756366; cupy-gpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | scipy-cpu 4.275e-07; torch-gpu 3.386e-07; cupy-gpu 3.386e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | numpy-cpu 3.259e-08; torch-gpu 3.386e-07; cupy-gpu 3.386e-07 |
| algos | maxabs-scaler | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | maxabs-scaler | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | maxabs-scaler | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | maxabs-scaler | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | minibatch-kmeans | istella | n_clusters | - | 8 | sklearn-cpu 8 |
| algos | minibatch-kmeans | istella | silhouette (higher is better) | - | 0.116696 | sklearn-cpu 0.111849 |
| algos | minibatch-kmeans | istella | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 0.622446 |
| algos | minibatch-kmeans | taxi | n_clusters | - | 8 | sklearn-cpu 8 |
| algos | minibatch-kmeans | taxi | silhouette (higher is better) | - | 0.138060 | sklearn-cpu 0.165473 |
| algos | minibatch-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | - | - | sklearn-cpu 0.525151 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.026193; torch-eager-tf32 1004.691291; torch-compile-tf32 1004.691291; torch-eager-bf16 22834.612745; torch-compile-bf16 22851.987745 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.548e-07; torch-eager-tf32 0.017534; torch-compile-tf32 0.017534; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | multinomial-nb | istella | accuracy (higher is better) | - | 0.853620 | sklearn-cpu 0.853620; cuml-gpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | - | 3.628565 | sklearn-cpu 3.087499; cuml-gpu 3.628599 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | - | 0.723160 | sklearn-cpu 0.723160; cuml-gpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | - | 0.590725 | sklearn-cpu 0.590725; cuml-gpu 0.590750 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067; cuml-gpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | sklearn-cpu 0.557319; cuml-gpu 0.559524 |
| algos | nadam | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.498e-07 |
| algos | nearest-centroid | istella | accuracy (higher is better) | - | 0.852610 | sklearn-cpu 0.852610 |
| algos | nearest-centroid | istella | logloss (lower is better) | - | 4.299485 | sklearn-cpu 4.117692 |
| algos | nearest-centroid | taxi | accuracy (higher is better) | - | 0.666700 | sklearn-cpu 0.666750 |
| algos | nearest-centroid | taxi | logloss (lower is better) | - | 0.781905 | sklearn-cpu 0.781690 |
| algos | normalizer | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 1.788e-07 |
| algos | normalizer | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 4.428e-08 |
| algos | normalizer | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 1.788e-07 |
| algos | normalizer | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 2.246e-08 |
| algos | onehot | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | onehot | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | onehot | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | onehot | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.438855 | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.150860 | statsforecast-cpu 49.356608 |
| algos | pagerank | istella | sum | - | 1.000000 | networkx-cpu 1.000000; cugraph-gpu 1.000000 |
| algos | pagerank | istella | l1_vs_networkx | - | - | networkx-cpu -; cugraph-gpu 0.010895 |
| algos | pagerank | taxi | sum | - | 1.000000 | networkx-cpu 1.000000; cugraph-gpu 1.000000 |
| algos | pagerank | taxi | l1_vs_networkx | - | - | networkx-cpu -; cugraph-gpu 0.007172 |
| algos | permutation-shap | istella | rel_error_vs_exact | - | 6.68e-08 | shap-cpu 3.692e-10; cuml-gpu 1.497e-07 |
| algos | permutation-shap | taxi | rel_error_vs_exact | - | 1.054e-07 | shap-cpu 1.144e-15; cuml-gpu 1.655e-07 |
| algos | poly-features | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | poly-features | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | poly-features | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | poly-features | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | power-transformer | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.063349 |
| algos | power-transformer | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.0003244 |
| algos | qr | istella | relative_gram_difference | - | 1.485e-07 | numpy-cpu 2.459e-08; torch-gpu 3.787e-07; cupy-gpu 3.787e-07 |
| algos | qr | taxi | relative_gram_difference | - | 1.714e-07 | numpy-cpu 3.024e-08; torch-gpu 5.971e-06; cupy-gpu 5.971e-06 |
| algos | quantile | istella | r2 (higher is better) | - | -0.039998 | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | - | 0.851877 | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | - | 0.899596 | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | - | 5.046749 | sklearn-cpu 5.044706 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | 0.0002359 | sklearn-cpu 0.0002359; torch-gpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | 0.027197 | sklearn-cpu 0.027197; torch-gpu 0.027197 |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.818182; torch-eager-tf32 1532.793045; torch-compile-tf32 1532.793045; torch-eager-bf16 17838.627100; torch-compile-bf16 18497.318029 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.381e-07; torch-eager-tf32 0.0003478; torch-compile-tf32 0.0003478; torch-eager-bf16 0.003621; torch-compile-bf16 0.003425 |
| algos | ridge-cv | istella | r2 (higher is better) | - | 0.328684 | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | 0.684422 | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | 0.908981 | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | 4.805109 | sklearn-cpu 4.805055 |
| algos | rmsprop | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | 0.967828 | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-tf32 0.953505; torch-compile-tf32 0.953505; torch-eager-bf16 0.953559; torch-compile-bf16 0.953559 |
| algos | rnn-clf | synthetic | logloss (lower is better) | - | 0.079029 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | 0.862684 | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-tf32 0.868056; torch-compile-tf32 0.868056; torch-eager-bf16 0.868001; torch-compile-bf16 0.868001 |
| algos | rnn-clf | taxi-hourly | logloss (lower is better) | - | 0.313008 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | rnn-reg | synthetic | r2 (higher is better) | - | 0.978602 | torch-eager-fp32 0.977347; torch-compile-fp32 0.977347; torch-eager-tf32 0.977347; torch-compile-tf32 0.977347; torch-eager-bf16 0.977345; torch-compile-bf16 0.977345 |
| algos | rnn-reg | synthetic | rmse (lower is better) | - | 0.169476 | torch-eager-fp32 0.174374; torch-compile-fp32 0.174374; torch-eager-tf32 0.174374; torch-compile-tf32 0.174374; torch-eager-bf16 0.174385; torch-compile-bf16 0.174385 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | 0.743904 | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-tf32 0.738800; torch-compile-tf32 0.738800; torch-eager-bf16 0.738906; torch-compile-bf16 0.738906 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | 0.548825 | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-tf32 0.554267; torch-compile-tf32 0.554267; torch-eager-bf16 0.554155; torch-compile-bf16 0.554155 |
| algos | robust-scaler | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | robust-scaler | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | robust-scaler | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | robust-scaler | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | sgd-clf | istella | accuracy (higher is better) | - | 0.920330 | sklearn-cpu 0.910200; cuml-gpu 0.809350 |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | 0.755330 | sklearn-cpu 0.752520; cuml-gpu 0.703120 |
| algos | sgd-reg | istella | r2 (higher is better) | - | 0.327829 | sklearn-cpu -2.197e+24; cuml-gpu 0.327768 |
| algos | sgd-reg | istella | rmse (lower is better) | - | 0.684858 | sklearn-cpu 1.238e+12; cuml-gpu 0.684889 |
| algos | sgd-reg | taxi | r2 (higher is better) | - | 0.908969 | sklearn-cpu 0.880681; cuml-gpu 0.908979 |
| algos | sgd-reg | taxi | rmse (lower is better) | - | 4.805411 | sklearn-cpu 5.501638; cuml-gpu 4.805168 |
| algos | simple-imputer | istella | masked_rmse | - | 346849.129968 | sklearn-cpu 346849.129968; cuml-gpu 346849.129968 |
| algos | simple-imputer | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | simple-imputer | taxi | masked_rmse | - | 5.985180 | sklearn-cpu 5.985180; cuml-gpu 5.985180 |
| algos | simple-imputer | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | sparse-rp | istella | mean_abs_distortion | - | 1.883381 | sklearn-cpu 0.474347; cuml-gpu 0.876709 |
| algos | sparse-rp | taxi | mean_abs_distortion | - | 0.147163 | sklearn-cpu 0.381016; cuml-gpu 0.266849 |
| algos | standard-scaler | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.022049 |
| algos | standard-scaler | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.0001523 |
| algos | standard-scaler | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.0003331 |
| algos | standard-scaler | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 3.216e-05 |
| algos | svd | istella | max_rel_singular_value_error | - | 662.224271 | numpy-cpu 1.000000; torch-gpu 7.758526; cupy-gpu 13326.362332 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | 3.379e-05 | numpy-cpu 4.1e-08; torch-gpu 2.861e-05; cupy-gpu 2.385e-06 |
| algos | svd | taxi | max_rel_singular_value_error | - | 7.036e-07 | numpy-cpu 4.308e-08; torch-gpu 2.595e-06; cupy-gpu 3.763e-06 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | 1.18e-06 | numpy-cpu 4.314e-08; torch-gpu 7.244e-06; cupy-gpu 6.886e-06 |
| algos | svgp | istella | r2 (higher is better) | - | -0.106016 | gpytorch-gpu -0.106040; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | 0.878373 | gpytorch-gpu 0.878383; gpytorch-cpu 0.878383 |
| algos | svgp | taxi | r2 (higher is better) | - | -0.194982 | gpytorch-gpu -0.209527; gpytorch-cpu -0.209546 |
| algos | svgp | taxi | rmse (lower is better) | - | 17.723482 | gpytorch-gpu 17.831018; gpytorch-cpu 17.831162 |
| algos | target-encoder | istella | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | target-encoder | istella | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | target-encoder | taxi | max_abs_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | target-encoder | taxi | rel_diff_vs_sklearn | - | - | sklearn-cpu -; cuml-gpu 0.000000 |
| algos | theta | synthetic | forecast_rmse (lower is better) | - | 1.436610 | statsforecast-cpu 1.436557; statsmodels-cpu 1.434862 |
| algos | theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.020604 | statsforecast-cpu 49.253901; statsmodels-cpu 49.311757 |
| algos | tree-shap | istella | max_additivity_error | - | 1.175e-06 | shap-cpu 1.837e-06; xgboost-cpu 1.837e-06; xgboost-gpu 1.956e-06; lightgbm-cpu 4.441e-15 |
| algos | tree-shap | taxi | max_additivity_error | - | 3.858e-05 | shap-cpu 0.0001201; xgboost-cpu 0.0001201; xgboost-gpu 5.402e-05; lightgbm-cpu 5.684e-13 |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.992124 | sklearn-cpu 0.992182; cuml-gpu 0.990033 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.998921 | sklearn-cpu 0.998823; cuml-gpu 0.998353 |
| classical | dbscan | istella | n_clusters | - | 40131 | cuml-gpu 40131 |
| classical | dbscan | istella | noise_fraction | - | 0.219391 | cuml-gpu 0.219391 |
| classical | dbscan | istella | rows | - | 1000000 | cuml-gpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | - | - | cuml-gpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | - | - | cuml-gpu 1.000000 |
| classical | dbscan | taxi | n_clusters | - | 36 | cuml-gpu 36 |
| classical | dbscan | taxi | noise_fraction | - | 0.000174 | cuml-gpu 0.000174 |
| classical | dbscan | taxi | rows | - | 1000000 | cuml-gpu 1000000 |
| classical | hdbscan | istella | n_clusters | - | 47 | cuml-gpu 53 |
| classical | hdbscan | istella | noise_fraction | - | 0.253810 | cuml-gpu 0.256280 |
| classical | hdbscan | istella | rows | - | 100000 | cuml-gpu 100000 |
| classical | hdbscan | taxi | n_clusters | - | 161 | cuml-gpu 159 |
| classical | hdbscan | taxi | noise_fraction | - | 0.140550 | cuml-gpu 0.130970 |
| classical | hdbscan | taxi | rows | - | 100000 | cuml-gpu 100000 |
| classical | kde | istella | mean_log_likelihood (higher is better) | - | -222.270583 | cuml-gpu -222.270582 |
| classical | kde | istella | rows_without_density | - | 0 | cuml-gpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | - | -14.826436 | cuml-gpu -14.826437 |
| classical | kde | taxi | rows_without_density | - | 0 | cuml-gpu 0 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | cuml-gpu 6.111e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | cuml-gpu 1.009914; torch-gpu 0.990156 |
| classical | kmeans | istella | n_iter | - | 33 | cuml-gpu 20; torch-gpu 55 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | cuml-gpu 3.093e+08; torch-gpu 3.129e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | cuml-gpu 1.000059; torch-gpu 1.011537 |
| classical | kmeans | taxi | n_iter | - | 91 | cuml-gpu 42; torch-gpu 85 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | cuml-gpu 0.976402; torch-gpu 0.981012 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | cuml-gpu 0; torch-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | cuml-gpu 0.999742; torch-gpu 0.999773 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | cuml-gpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | cuml-gpu -11031.855105; torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | cuml-gpu 87.647429; torch-gpu nan; torch-gpu-eigh 0.768590 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908837 | cuml-gpu 0.908836; torch-gpu 0.908836; torch-gpu-eigh 0.908836 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696477 | cuml-gpu 4.696488; torch-gpu 4.696480; torch-gpu-eigh 4.696490 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | torch-gpu 1.000000; cuml-gpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999997 | torch-gpu 0.999996; cuml-gpu 0.999996 |
| classical | svc | istella | accuracy (higher is better) | - | 0.922200 | cuml-gpu 0.922200 |
| classical | svc | istella | n_support | - | 2400 | cuml-gpu 2401 |
| classical | svc | taxi | accuracy (higher is better) | - | 0.767500 | cuml-gpu 0.767500 |
| classical | svc | taxi | n_support | - | 5527 | cuml-gpu 5541 |
| classical2 | elasticnet | istella | r2 (higher is better) | - | 0.260922 | cuml-gpu 0.260922 |
| classical2 | elasticnet | istella | rmse (lower is better) | - | 0.718134 | cuml-gpu 0.718134 |
| classical2 | elasticnet | taxi | r2 (higher is better) | - | 0.907378 | cuml-gpu 0.907378 |
| classical2 | elasticnet | taxi | rmse (lower is better) | - | 4.847225 | cuml-gpu 4.847224 |
| classical2 | gmm | istella | bic (lower is better) | - | -3.851e+07 | - |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | - | 200.794500 | - |
| classical2 | gmm | istella | n_iter | - | 24 | - |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.668e+06 | - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.807640 | - |
| classical2 | gmm | taxi | n_iter | - | 29 | - |
| classical2 | ivf | istella | recall_at_k (higher is better) | - | 0.999925 | cuvs-gpu - |
| classical2 | ivf | istella | rows_with_repeated_ids | - | 0 | cuvs-gpu - |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | 0.999650 | cuvs-gpu - |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | 0 | cuvs-gpu - |
| classical2 | lasso | istella | r2 (higher is better) | - | 0.310837 | cuml-gpu 0.310837 |
| classical2 | lasso | istella | rmse (lower is better) | - | 0.693460 | cuml-gpu 0.693460 |
| classical2 | lasso | taxi | r2 (higher is better) | - | 0.908995 | cuml-gpu 0.908995 |
| classical2 | lasso | taxi | rmse (lower is better) | - | 4.804744 | cuml-gpu 4.804745 |
| classical2 | logreg | istella | accuracy (higher is better) | - | 0.924590 | cuml-gpu 0.924550 |
| classical2 | logreg | istella | logloss (lower is better) | - | 0.181249 | cuml-gpu 0.181262 |
| classical2 | logreg | istella | nonfinite_proba_rows | - | 0 | cuml-gpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | - | 0.763350 | cuml-gpu 0.763340 |
| classical2 | logreg | taxi | logloss (lower is better) | - | 0.538985 | cuml-gpu 0.538984 |
| classical2 | logreg | taxi | nonfinite_proba_rows | - | 0 | cuml-gpu 0 |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328674 | cuml-gpu -0.251259 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684427 | cuml-gpu 0.934403 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | cuml-gpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805050 | cuml-gpu 4.805051 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | cuml-gpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | - | 0.0001314 | cuml-gpu 0.0001472 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999965 | cuml-gpu 0.999964 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | - | 0.003257 | cuml-gpu 0.003257 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | torch-eager-fp32 1.401e-06; torch-eager-tf32 0.0002784; torch-compile-fp32 1.401e-06; torch-compile-tf32 0.0002784; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 0.0005341; torch-eager-tf32 0.100822; torch-compile-fp32 0.0005341; torch-compile-tf32 0.100822; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.475e-06; torch-eager-tf32 0.0002785; torch-compile-fp32 1.475e-06; torch-compile-tf32 0.0002785; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-tf32 9.018733; torch-compile-fp32 9.018733; torch-compile-tf32 9.018732; torch-eager-bf16 9.018664; torch-compile-bf16 9.018669 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.371e-06; torch-eager-tf32 0.0007233; torch-compile-fp32 1.341e-06; torch-compile-tf32 0.0007184; torch-eager-bf16 0.006870; torch-compile-bf16 0.006304 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.882e-06; torch-eager-tf32 0.0009929; torch-compile-fp32 1.841e-06; torch-compile-tf32 0.0009862; torch-eager-bf16 0.009431; torch-compile-bf16 0.008653 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-tf32 9.018732; torch-compile-fp32 9.018734; torch-compile-tf32 9.018732; torch-eager-bf16 9.018402; torch-compile-bf16 9.018669 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.418446 | torch-eager-fp32 8.422415; torch-eager-tf32 8.422506; torch-compile-fp32 8.422415; torch-compile-tf32 8.422516; torch-eager-bf16 8.421570; torch-compile-bf16 8.420250 |
| neural | lm-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | lm-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 0.000000; torch-eager-tf32 9.537e-07; torch-compile-fp32 9.537e-07; torch-compile-tf32 9.537e-07; torch-eager-bf16 0.0003309; torch-compile-bf16 6.39e-05 |
| neural | lm-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 3.815e-06; torch-eager-tf32 9.537e-05; torch-compile-fp32 3.815e-06; torch-compile-tf32 0.0001049; torch-eager-bf16 0.0008411; torch-compile-bf16 0.002161 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | mamba-ssm-fp32 1.192e-07; mamba-ssm-tf32 3.815e-06; torch-eager-bf16 5.15e-05; torch-eager-fp32 1.192e-07; torch-eager-tf32 3.815e-06 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | mamba-ssm-fp32 5.945e-08; mamba-ssm-tf32 1.902e-06; torch-eager-bf16 2.568e-05; torch-eager-fp32 5.945e-08; torch-eager-tf32 1.902e-06 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | mamba-ssm-fp32 1.907e-06; mamba-ssm-tf32 0.0005126; torch-compile-bf16 0.007544; torch-compile-fp32 2.146e-06; torch-compile-tf32 0.000509; torch-eager-bf16 0.007544; torch-eager-fp32 2.146e-06; torch-eager-tf32 0.000509 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | mamba-ssm-fp32 6.485e-07; mamba-ssm-tf32 0.0001743; torch-compile-bf16 0.002565; torch-compile-fp32 7.295e-07; torch-compile-tf32 0.0001731; torch-eager-bf16 0.002565; torch-eager-fp32 7.295e-07; torch-eager-tf32 0.0001731 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | mamba-ssm-fp32 0.001235; mamba-ssm-tf32 0.001376; torch-compile-bf16 0.001982; torch-compile-fp32 4.768e-07; torch-compile-tf32 0.0001798; torch-eager-bf16 0.002095; torch-eager-fp32 4.768e-07; torch-eager-tf32 0.0001827 |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | mamba-ssm-fp32 0.0005401; mamba-ssm-tf32 0.0006018; torch-compile-bf16 0.0008672; torch-compile-fp32 2.086e-07; torch-compile-tf32 7.867e-05; torch-eager-bf16 0.0009164; torch-eager-fp32 2.086e-07; torch-eager-tf32 7.993e-05 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | torch-eager-fp32 1.160401; torch-eager-tf32 1.160392; torch-compile-fp32 1.160401; torch-compile-tf32 1.160392; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | torch-eager-fp32 1.123361; torch-eager-tf32 1.123355; torch-compile-fp32 1.123361; torch-compile-tf32 1.123355; torch-eager-bf16 1.123461; torch-compile-bf16 1.123462 |
| neural | mlp-train-step | gaussian | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 2.384e-07; torch-eager-tf32 8.941e-06; torch-compile-fp32 2.384e-07; torch-compile-tf32 8.941e-06; torch-eager-bf16 9.656e-05; torch-compile-bf16 9.656e-05 |
| neural | mlp-train-step | gaussian | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-eager-fp32 2.384e-07; torch-eager-tf32 5.484e-06; torch-compile-fp32 2.384e-07; torch-compile-tf32 5.603e-06; torch-eager-bf16 0.0001006; torch-compile-bf16 0.0001007 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | torch-eager-fp32 5.635910; torch-eager-tf32 5.635948; torch-compile-fp32 5.635910; torch-compile-tf32 5.635956; torch-eager-bf16 5.635952; torch-compile-bf16 5.635985 |
| neural | samba-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 2.235e-06; torch-eager-tf32 0.001577; torch-compile-fp32 2.056e-06; torch-compile-tf32 0.001510; torch-eager-bf16 0.019621; torch-compile-bf16 0.016407 |
| neural | samba-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.257e-06; torch-eager-tf32 0.0008869; torch-compile-fp32 1.156e-06; torch-compile-tf32 0.0008487; torch-eager-bf16 0.011031; torch-compile-bf16 0.009224 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | torch-compile-bf16 -; torch-compile-fp32 5.635910; torch-compile-tf32 5.635948; torch-eager-bf16 5.635952; torch-eager-fp32 5.635910; torch-eager-tf32 5.635948 |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | torch-compile-bf16 -; torch-compile-fp32 4.833934; torch-compile-tf32 4.833748; torch-eager-bf16 4.833967; torch-eager-fp32 4.833934; torch-eager-tf32 4.833591 |
| neural | samba-train-step | bytes | steps | - | 2 | torch-compile-bf16 -; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-eager-fp32 2; torch-eager-tf32 2 |
| neural | samba-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-compile-bf16 -; torch-compile-fp32 0.000000; torch-compile-tf32 3.815e-05; torch-eager-bf16 4.196e-05; torch-eager-fp32 4.768e-07; torch-eager-tf32 3.815e-05 |
| neural | samba-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | torch-compile-bf16 -; torch-compile-fp32 4.768e-07; torch-compile-tf32 0.0001855; torch-eager-bf16 3.338e-05; torch-eager-fp32 4.768e-07; torch-eager-tf32 0.0003433 |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 4.768e-07; torch-eager-tf32 0.0002124; torch-compile-fp32 4.768e-07; torch-compile-tf32 0.0002124; torch-eager-bf16 0.001819; torch-compile-bf16 0.001819 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 1.02e-07; torch-eager-tf32 4.545e-05; torch-compile-fp32 1.02e-07; torch-compile-tf32 4.545e-05; torch-eager-bf16 0.0003893; torch-compile-bf16 0.0003893 |
| trees | et | istella | auc (higher is better) | - | 0.937987 | sklearn-et-cpu 0.937659 |
| trees | et | istella | logloss (lower is better) | - | 0.189989 | sklearn-et-cpu 0.190177 |
| trees | et | taxi | auc (higher is better) | - | 0.618907 | sklearn-et-cpu 0.619262 |
| trees | et | taxi | logloss (lower is better) | - | 0.526142 | sklearn-et-cpu 0.525951 |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | 0.631297 | catboost-gpu 0.630433; xgboost-gpu 0.631978; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | 0.528285 | catboost-gpu 0.528509; xgboost-gpu 0.528548; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.983304 | catboost-gpu 0.983291; xgboost-gpu 0.983622; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.156072 | catboost-gpu 0.155812; xgboost-gpu 0.149263; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.632205 | catboost-gpu 0.632248; xgboost-gpu 0.630969; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.527838 | catboost-gpu 0.528018; xgboost-gpu 0.528678; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983586 | catboost-gpu 0.983599; xgboost-gpu 0.983622; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.150066 | catboost-gpu 0.149896; xgboost-gpu 0.149263; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.632096 | catboost-gpu 0.632012; xgboost-gpu 0.630969; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528044 | catboost-gpu 0.528025; xgboost-gpu 0.528678; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.907556 | catboost-gpu 0.907846; xgboost-gpu 0.910140; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.258413 | catboost-gpu 0.258180; xgboost-gpu 0.246803; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.599380 | catboost-gpu 0.599270; xgboost-gpu 0.601200; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.012595 | catboost-gpu 1.012704; xgboost-gpu 1.005204; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.979444 | catboost-gpu 0.979541; catboost-cpu - |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.190928 | catboost-gpu 0.190146; catboost-cpu - |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.629203 | catboost-gpu 0.629073; catboost-cpu - |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.529007 | catboost-gpu 0.528987; catboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.854545 | catboost-gpu 0.853929; xgboost-gpu 0.872796; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.719953 | catboost-gpu 0.719621; xgboost-gpu 0.738397; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.650400 | catboost-gpu 0.650025; xgboost-gpu 0.670093; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-gpu 0.980020; catboost-cpu - |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-gpu 0.187222; catboost-cpu - |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-gpu 0.630348; catboost-cpu - |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-gpu 0.528586; catboost-cpu - |
| trees | rf | istella | auc (higher is better) | - | 0.945385 | cuml-rf-gpu 0.945348 |
| trees | rf | istella | logloss (lower is better) | - | 0.182017 | cuml-rf-gpu 0.182069 |
| trees | rf | taxi | auc (higher is better) | - | 0.617838 | cuml-rf-gpu 0.617836 |
| trees | rf | taxi | logloss (lower is better) | - | 0.525953 | cuml-rf-gpu 0.525954 |


## Neural headline: ours IDENTICAL against torch bf16

What customers run is torch in bf16. The headline divides our IDENTICAL median by torch's fastest bf16 arm (eager or compile, the lower stored median); the fp32 twin (torch's fastest fp32 arm) is the second column. Per-arm ratios stay in each race's table below.

| lane | dataset | ours IDENTICAL ms | torch bf16 ms (fastest arm) | ours / torch bf16 | torch fp32 twin ms (fastest arm) | ours / torch fp32 | note |
|---|---|---|---|---|---|---|---|
| adafactor | synthetic | 13.7 | - | - | 15.9 (torch-eager-fp32) | 0.863 | no torch bf16 arm on this lane |
| adagrad | synthetic | 8.5 | - | - | 14.8 (torch-eager-fp32) | 0.570 | no torch bf16 arm on this lane |
| adamax | synthetic | 11.2 | - | - | 18.4 (torch-eager-fp32) | 0.606 | no torch bf16 arm on this lane |
| avgpool1d | synthetic | 1.0 | 1.1 (torch-eager-bf16) | 0.885 | 0.9 (torch-compile-fp32) | 1.114 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| avgpool2d | synthetic | 4.5 | 1.6 (torch-eager-bf16) | 2.898 | 1.4 (torch-compile-fp32) | 3.182 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm1d | synthetic | 1.6 | 1.6 (torch-compile-bf16) | 0.989 | 1.8 (torch-compile-fp32) | 0.889 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm2d | synthetic | 1.3 | 1.5 (torch-eager-bf16) | 0.836 | 1.4 (torch-compile-fp32) | 0.879 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| cnn-clf | synthetic | 200.4 | 302.4 (torch-eager-bf16) | 0.663 | 208.8 (torch-eager-fp32) | 0.959 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv1d | synthetic | 10.0 | 3.3 (torch-eager-bf16) | 3.085 | 3.9 (torch-eager-fp32) | 2.585 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv2d | synthetic | 6.3 | 1.5 (torch-eager-bf16) | 4.073 | 2.3 (torch-eager-fp32) | 2.774 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| dropout2d | synthetic | 1.1 | - | - | 1.4 (torch-compile-fp32) | 0.772 | no torch bf16 arm on this lane |
| embedding | synthetic | 2.7 | - | - | 2.1 (torch-compile-fp32) | 1.290 | no torch bf16 arm on this lane |
| gcn | istella | 3.0 | 5.3 (torch-compile-bf16) | 0.571 | 3.8 (torch-compile-fp32) | 0.804 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gcn | taxi | 3.0 | 8.8 (torch-compile-bf16) | 0.336 | 3.2 (torch-compile-fp32) | 0.927 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-avgpool | synthetic | 2.4 | 0.9 (torch-eager-bf16) | 2.733 | 0.6 (torch-eager-fp32) | 4.212 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-maxpool | synthetic | 3.2 | 1.2 (torch-compile-bf16) | 2.648 | 0.6 (torch-eager-fp32) | 5.409 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | istella | 7.5 | 3.5 (torch-compile-bf16) | 2.159 | 3.9 (torch-compile-fp32) | 1.925 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | taxi | 1.7 | 1.8 (torch-compile-bf16) | 0.978 | 1.4 (torch-eager-fp32) | 1.253 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | synthetic | 447.1 | 717.1 (torch-compile-bf16) | 0.623 | 555.9 (torch-compile-fp32) | 0.804 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | taxi-hourly | 447.7 | 733.2 (torch-compile-bf16) | 0.611 | 552.2 (torch-eager-fp32) | 0.811 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | synthetic | 446.3 | 819.8 (torch-compile-bf16) | 0.544 | 538.4 (torch-eager-fp32) | 0.829 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | taxi-hourly | 447.0 | 754.4 (torch-eager-bf16) | 0.593 | 626.5 (torch-eager-fp32) | 0.713 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lamb | synthetic | 23.8 | - | - | - | - | no torch bf16 arm on this lane |
| layernorm | synthetic | 1.1 | 1.5 (torch-eager-bf16) | 0.732 | 1.6 (torch-eager-fp32) | 0.715 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lion | synthetic | 11.0 | - | - | - | - | no torch bf16 arm on this lane |
| lstm-clf | synthetic | 546.6 | 735.7 (torch-eager-bf16) | 0.743 | 539.4 (torch-eager-fp32) | 1.013 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-clf | taxi-hourly | 546.5 | 707.9 (torch-eager-bf16) | 0.772 | 531.6 (torch-eager-fp32) | 1.028 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | synthetic | 545.6 | 729.3 (torch-eager-bf16) | 0.748 | 574.8 (torch-compile-fp32) | 0.949 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | taxi-hourly | 545.8 | 742.3 (torch-eager-bf16) | 0.735 | 606.4 (torch-compile-fp32) | 0.900 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool1d | synthetic | 1.2 | 1.7 (torch-compile-bf16) | 0.741 | 1.2 (torch-eager-fp32) | 1.020 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool2d | synthetic | 1.5 | 1.9 (torch-eager-bf16) | 0.775 | 1.9 (torch-eager-fp32) | 0.757 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| moe | synthetic | 14.1 | 4.5 (torch-eager-bf16) | 3.157 | 11.9 (torch-compile-fp32) | 1.190 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| nadam | synthetic | 10.8 | - | - | 21.8 (torch-eager-fp32) | 0.496 | no torch bf16 arm on this lane |
| resnet-block | synthetic | 15.3 | 2.7 (torch-compile-bf16) | 5.756 | 4.8 (torch-compile-fp32) | 3.203 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rmsprop | synthetic | 8.6 | - | - | 14.8 (torch-eager-fp32) | 0.579 | no torch bf16 arm on this lane |
| rnn-clf | synthetic | 244.0 | 829.7 (torch-eager-bf16) | 0.294 | 497.4 (torch-eager-fp32) | 0.491 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-clf | taxi-hourly | 243.9 | 687.6 (torch-compile-bf16) | 0.355 | 550.3 (torch-compile-fp32) | 0.443 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | synthetic | 242.4 | 653.6 (torch-eager-bf16) | 0.371 | 456.3 (torch-eager-fp32) | 0.531 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | taxi-hourly | 242.4 | 648.1 (torch-compile-bf16) | 0.374 | 570.0 (torch-eager-fp32) | 0.425 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gemm | gaussian | 52.6 | 47.0 (torch-eager-bf16) | 1.119 | 51.0 (torch-eager-fp32) | 1.032 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-forward | bytes | 55.4 | 49.4 (torch-eager-bf16) | 1.122 | 48.5 (torch-compile-fp32) | 1.143 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-train-step | bytes | 37.5 | 11.0 (torch-compile-bf16) | 3.420 | 19.5 (torch-compile-fp32) | 1.922 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba1-forward | gaussian | 5.4 | 195.9 (torch-eager-bf16) | 0.028 | 149.6 (torch-eager-fp32) | 0.036 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba2-forward | gaussian | 7.8 | 21.7 (torch-eager-bf16) | 0.361 | 25.8 (torch-eager-fp32) | 0.304 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba3-forward | gaussian | 4.1 | 9.1 (torch-compile-bf16) | 0.451 | 13.5 (torch-compile-fp32) | 0.303 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mlp-train-step | gaussian | 2.0 | 2.0 (torch-eager-bf16) | 0.994 | 1.8 (torch-eager-fp32) | 1.132 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-forward | bytes | 5.0 | 5.3 (torch-compile-bf16) | 0.937 | 6.0 (torch-compile-fp32) | 0.836 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-train-step | bytes | 42.4 | 157.5 (torch-eager-bf16) | 0.269 | 57.5 (torch-compile-fp32) | 0.736 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| transformer-forward | gaussian | 7.6 | 3.6 (torch-eager-bf16) | 2.109 | 3.8 (torch-eager-fp32) | 2.016 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |

## Trees

### et / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 4917.3 | 4917.3..4917.3 | 1 | - | - | 4917.3 | - | - (stored whole) | - | - | - | - | auc=0.937987, logloss=0.189989 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-et-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 16539.7 | 16539.7..16539.7 | 1 | 0.297 | - | 16539.7 | 16539.7 | 0.00 (cpu-arm) | 0.297 (whole/whole) | - | 24890.4 | - | auc=0.937659, logloss=0.190177 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-et-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### et / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 4359.3 | 4359.3..4359.3 | 1 | - | - | 4359.3 | - | - (stored whole) | - | - | - | - | auc=0.618907, logloss=0.526142 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-et-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 12501.5 | 12501.5..12501.5 | 1 | 0.349 | - | 12501.5 | 12501.5 | 0.00 (cpu-arm) | 0.349 (whole/whole) | - | 13627.4 | - | auc=0.619262, logloss=0.525951 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-et-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0112)`, ran on NVIDIA L40S (nv2, RunPod) job v0112

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 7305.0 | 7305.0..7305.0 | 1 | - | - | 7305.0 | - | - (stored whole) | - | - | - | - | auc=0.631297, logloss=0.528285 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0112 2026-10-11; identity vs amd-mi325x: MATCH) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 26094.5 | 26094.5..26094.5 | 1 | 0.280 | - | 26094.5 | - | - (stored whole) | 0.280 (whole/whole) | - | 6472.1 | 1524.0 | auc=0.630433, logloss=0.528509 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 13560.4 | 13560.4..13560.4 | 1 | 0.539 | - | 13560.4 | - | - (stored whole) | 0.539 (whole/whole) | - | 4475.6 | 1524.0 | auc=0.631978, logloss=0.528548 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host f003873fb257) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 7232.6 | 7232.6..7232.6 | 1 | - | - | 7232.6 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 9362.5 | 9362.5..9362.5 | 1 | 0.773 | - | 9362.5 | - | - (stored whole) | 0.773 (whole/whole) | - | 7027.2 | 1270.0 | auc=0.983291, logloss=0.155812 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 8075.5 | 8075.5..8075.5 | 1 | 0.896 | - | 8075.5 | - | - (stored whole) | 0.896 (whole/whole) | - | 8157.7 | 1270.0 | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 3692.4 | 3692.4..3692.4 | 1 | - | - | 3692.4 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 6485.3 | 6485.3..6485.3 | 1 | 0.569 | - | 6485.3 | - | - (stored whole) | 0.569 (whole/whole) | - | 2961.0 | 1266.0 | auc=0.632248, logloss=0.528018 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 3249.7 | 3249.7..3249.7 | 1 | 1.136 | - | 3249.7 | - | - (stored whole) | 1.136 (whole/whole) | - | 2821.7 | 1266.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0112)`, ran on NVIDIA L40S (nv2, RunPod) job v0112

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 7437.9 | 7437.9..7437.9 | 1 | - | - | 7437.9 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0112 2026-10-11; identity vs amd-mi325x: MATCH) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 22265.6 | 22265.6..22265.6 | 1 | 0.334 | - | 22265.6 | - | - (stored whole) | 0.334 (whole/whole) | - | 7621.2 | 1266.0 | auc=0.983599, logloss=0.149896 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 11997.7 | 11997.7..11997.7 | 1 | 0.620 | - | 11997.7 | - | - (stored whole) | 0.620 (whole/whole) | - | 8630.0 | 1266.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host f003873fb257) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0112)`, ran on NVIDIA L40S (nv2, RunPod) job v0112

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3743.9 | 3743.9..3743.9 | 1 | - | - | 3743.9 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0112 2026-10-11; identity vs amd-mi325x: MATCH) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 15946.6 | 15946.6..15946.6 | 1 | 0.235 | - | 15946.6 | - | - (stored whole) | 0.235 (whole/whole) | - | 2967.1 | 1266.0 | auc=0.632012, logloss=0.528025 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 6550.6 | 6550.6..6550.6 | 1 | 0.572 | - | 6550.6 | - | - (stored whole) | 0.572 (whole/whole) | - | 2893.9 | 1266.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host f003873fb257) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 14667.8 | 14667.8..14667.8 | 1 | - | - | 14667.8 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 14789.2 | 14789.2..14789.2 | 1 | 0.992 | - | 14789.2 | - | - (stored whole) | 0.992 (whole/whole) | - | 7636.1 | 1582.0 | accuracy=0.907846, mlogloss=0.258180 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 32408.6 | 32408.6..32408.6 | 1 | 0.453 | - | 32408.6 | - | - (stored whole) | 0.453 (whole/whole) | - | 8690.1 | 1582.0 | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host f003873fb257) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 7044.5 | 7044.5..7044.5 | 1 | - | - | 7044.5 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 10105.2 | 10105.2..10105.2 | 1 | 0.697 | - | 10105.2 | - | - (stored whole) | 0.697 (whole/whole) | - | 3052.7 | 1378.0 | accuracy=0.599270, mlogloss=1.012704 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 13072.6 | 13072.6..13072.6 | 1 | 0.539 | - | 13072.6 | - | - (stored whole) | 0.539 (whole/whole) | - | 3284.4 | 1378.0 | accuracy=0.601200, mlogloss=1.005204 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host f003873fb257) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 35370.0 | 35370.0..35370.0 | 1 | - | - | 35370.0 | - | - (stored whole) | - | - | - | - | auc=0.979444, logloss=0.190928 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 38076.0 | 38076.0..38076.0 | 1 | 0.929 | - | 38076.0 | - | - (stored whole) | 0.929 (whole/whole) | - | 6338.1 | 4782.0 | auc=0.979541, logloss=0.190146 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 19986.4 | 19986.4..19986.4 | 1 | - | - | 19986.4 | - | - (stored whole) | - | - | - | - | auc=0.629203, logloss=0.529007 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 20726.1 | 20726.1..20726.1 | 1 | 0.964 | - | 20726.1 | - | - (stored whole) | 0.964 (whole/whole) | - | 2633.1 | 7854.0 | auc=0.629073, logloss=0.528987 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 2768.8 | 2768.8..2768.8 | 1 | - | - | 2768.8 | - | - (stored whole) | - | - | - | - | map=0.854545, ndcg10=0.719953, ndcg5=0.650400 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 3806.7 | 3806.7..3806.7 | 1 | 0.727 | - | 3806.7 | - | - (stored whole) | 0.727 (whole/whole) | - | 8509.7 | 1836.0 | map=0.853929, ndcg10=0.719621, ndcg5=0.650025 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 3806.7 | 3806.7..3806.7 | 1 | 0.727 | - | 3806.7 | - | - (stored whole) | 0.727 (whole/whole) | - | 9099.1 | 1836.0 | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 6137.4 | 6137.4..6137.4 | 1 | - | - | 6137.4 | - | - (stored whole) | - | - | - | - | auc=0.980129, logloss=0.186686 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 7580.6 | 7580.6..7580.6 | 1 | 0.810 | - | 7580.6 | - | - (stored whole) | 0.810 (whole/whole) | - | 6320.6 | 1198.0 | auc=0.980020, logloss=0.187222 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 3503.2 | 3503.2..3503.2 | 1 | - | - | 3503.2 | - | - (stored whole) | - | - | - | - | auc=0.630436, logloss=0.528554 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host f003873fb257) | catboost 1.2.10 | opponent | 5721.3 | 5721.3..5721.3 | 1 | 0.612 | - | 5721.3 | - | - (stored whole) | 0.612 (whole/whole) | - | 2467.7 | 940.0 | auc=0.630348, logloss=0.528586 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rf / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 30726.6 | 30726.6..30726.6 | 1 | - | - | 30726.6 | - | - (stored whole) | - | - | - | - | auc=0.945385, logloss=0.182017 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| cuml-rf-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 24051.6 | 24051.6..24051.6 | 1 | 1.278 | - | 24051.6 | - | - (stored whole) | 1.278 (whole/whole) | - | 21698.2 | 3260.0 | auc=0.945348, logloss=0.182069 | yes | UNKNOWN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rf / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 9400.8 | 9400.8..9400.8 | 1 | - | - | 9400.8 | - | - (stored whole) | - | - | - | - | auc=0.617838, logloss=0.525953 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| cuml-rf-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 21000.3 | 21000.3..21000.3 | 1 | 0.448 | - | 21000.3 | - | - (stored whole) | 0.448 (whole/whole) | - | 3056.4 | 1724.0 | auc=0.617836, logloss=0.525954 | yes | UNKNOWN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0011)`, ran on NVIDIA L40S (nv2, RunPod) job v0011

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 85163.3 | 85163.3..85163.3 | 1 | - | - | 85163.3 | - | - (stored whole) | - | - | - | - | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | - | main board, one scored run | - | ok (main@5876e9205 nv2/v0011 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 66379.1 | 66379.1..66379.1 | 1 | 1.283 | - | 67048.6 | 66379.1 | 669.57 (upload_ms_untimed) | 1.270 (whole/whole) | - | 2599.0 | 1272.0 | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0011)`, ran on NVIDIA L40S (nv2, RunPod) job v0011

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 13203.7 | 13203.7..13203.7 | 1 | - | - | 13203.7 | - | - (stored whole) | - | - | - | - | n_clusters=36, noise_fraction=0.000174, rows=1000000 | - | main board, one scored run | - | ok (main@5876e9205 nv2/v0011 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 440431.5 | 440431.5..440431.5 | 1 | 0.030 | - | 440479.4 | 440431.5 | 47.88 (upload_ms_untimed) | 0.030 (whole/whole) | - | 850.3 | 474.0 | n_clusters=36, noise_fraction=0.000174, rows=1000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0132)`, ran on NVIDIA L40S (nv2, RunPod) job v0132

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1193.1 | 1193.1..1193.1 | 1 | - | - | 1193.1 | - | - (stored whole) | - | - | - | - | n_clusters=47, noise_fraction=0.253810, rows=100000 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0132 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 862.1 | 862.1..862.1 | 1 | 1.384 | - | 959.5 | 862.1 | 97.42 (upload_ms_untimed) | 1.243 (whole/whole) | - | 1864.3 | 526.0 | n_clusters=53, noise_fraction=0.256280, rows=100000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0132)`, ran on NVIDIA L40S (nv2, RunPod) job v0132

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 426.0 | 426.0..426.0 | 1 | - | - | 426.0 | - | - (stored whole) | - | - | - | - | n_clusters=161, noise_fraction=0.140550, rows=100000 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0132 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 371.2 | 371.2..371.2 | 1 | 1.148 | - | 383.0 | 371.2 | 11.75 (upload_ms_untimed) | 1.112 (whole/whole) | - | 969.5 | 448.0 | n_clusters=159, noise_fraction=0.130970, rows=100000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kde / istella (rows full, shape 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0122)`, ran on NVIDIA L40S (nv2, RunPod) job v0122

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 27.7 | 27.7..27.7 | 1 | - | - | 27.7 | - | - (stored whole) | - | - | - | - | mean_log_likelihood=-222.270583, rows_without_density=0 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0122 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 6.9 | 6.9..6.9 | 1 | 4.023 | - | 116.8 | 6.9 | 109.95 (upload_ms_untimed) | 0.237 (whole/whole) | - | 1001.5 | 514.0 | mean_log_likelihood=-222.270582, rows_without_density=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0122)`, ran on NVIDIA L40S (nv2, RunPod) job v0122

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 18.8 | 18.8..18.8 | 1 | - | - | 18.8 | - | - (stored whole) | - | - | - | - | mean_log_likelihood=-14.826436, rows_without_density=0 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0122 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1.9 | 1.9..1.9 | 1 | 9.945 | - | 13.0 | 1.9 | 11.14 (upload_ms_untimed) | 1.440 (whole/whole) | - | 800.2 | 436.0 | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0010)`, ran on NVIDIA L40S (nv2, RunPod) job v0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 483.5 | 483.5..483.5 | 1 | - | - | 483.5 | - | - (stored whole) | - | - | - | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | - | main board, one scored run | - | ok (main@5876e9205 nv2/v0010 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 483.4 | 483.4..483.4 | 1 | 1.000 | - | 2077.7 | 483.4 | 1594.30 (upload_ms_untimed) | 0.233 (whole/whole) | - | 4957.5 | 2154.0 | inertia=6.111e+17, inertia_over_ours=1.009914, n_iter=20 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 857.6 | 857.6..857.6 | 1 | 0.564 | - | 950.9 | 857.6 | 93.34 (upload_ms_untimed) | 0.508 (whole/whole (kernel not derivable)) | - | 3168.9 | 3468.9 | inertia=5.991e+17, inertia_over_ours=0.990156, n_iter=55 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0010)`, ran on NVIDIA L40S (nv2, RunPod) job v0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 187.3 | 187.3..187.3 | 1 | - | - | 187.3 | - | - (stored whole) | - | - | - | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | - | main board, one scored run | - | ok (main@5876e9205 nv2/v0010 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 390.3 | 390.3..390.3 | 1 | 0.480 | - | 572.6 | 390.3 | 182.28 (upload_ms_untimed) | 0.327 (whole/whole) | - | 1227.3 | 614.0 | inertia=3.093e+08, inertia_over_ours=1.000059, n_iter=42 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 891.5 | 891.5..891.5 | 1 | 0.210 | - | 900.1 | 891.5 | 8.59 (upload_ms_untimed) | 0.208 (whole/whole (kernel not derivable)) | - | 1254.7 | 438.0 | inertia=3.129e+08, inertia_over_ours=1.011537, n_iter=85 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn / istella (rows full, shape 400000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0010)`, ran on NVIDIA L40S (nv2, RunPod) job v0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 45.2 | 45.2..45.2 | 1 | - | - | 45.2 | - | - (stored whole) | - | - | - | - | recall_at_k=0.976250, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@5876e9205 nv2/v0010 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 73.2 | 73.2..73.2 | 1 | 0.618 | - | 475.8 | 73.2 | 402.63 (upload_ms_untimed) | 0.095 (whole/whole) | - | 1623.0 | 772.0 | recall_at_k=0.976402, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 115.5 | 115.5..115.5 | 1 | 0.392 | - | 137.6 | 115.5 | 22.16 (upload_ms_untimed) | 0.329 (whole/whole (kernel not derivable)) | - | 1219.7 | 3816.8 | recall_at_k=0.981012, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0010)`, ran on NVIDIA L40S (nv2, RunPod) job v0010

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 24.6 | 24.6..24.6 | 1 | - | - | 24.6 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999754, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@5876e9205 nv2/v0010 2026-10-11; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 40.3 | 40.3..40.3 | 1 | 0.609 | - | 70.6 | 40.3 | 30.31 (upload_ms_untimed) | 0.348 (whole/whole) | - | 818.7 | 452.0 | recall_at_k=0.999742, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 96.4 | 96.4..96.4 | 1 | 0.255 | - | 98.3 | 96.4 | 1.90 (upload_ms_untimed) | 0.250 (whole/whole (kernel not derivable)) | - | 895.3 | 3174.8 | recall_at_k=0.999773, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 504.6 | 504.6..504.6 | 1 | - | - | 504.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 79.4 | 79.4..79.4 | 1 | 6.352 | - | 1371.1 | 79.4 | 1291.62 (upload_ms_untimed) | 0.368 (whole/whole) | - | 5076.3 | 2154.0 | finite=True, r2=-11031.855105, rmse=87.647429 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 582.0 | 582.0..582.0 | 1 | 0.867 | - | 728.7 | 582.0 | 146.67 (upload_ms_untimed) | 0.692 (whole/whole (kernel not derivable)) | - | 3013.1 | 8894.4 | finite=False, r2=nan, rmse=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu-eigh | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 19.9 | 19.9..19.9 | 1 | 25.309 | - | 166.7 | 19.9 | 146.78 (upload_ms_untimed) | 3.026 (whole/whole (kernel not derivable)) | - | 3044.6 | 3455.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 14.9 | 14.9..14.9 | 1 | - | - | 14.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908837, rmse=4.696477 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 22.7 | 22.7..22.7 | 1 | 0.659 | - | 241.7 | 22.7 | 219.05 (upload_ms_untimed) | 0.062 (whole/whole) | - | 1380.9 | 614.0 | finite=True, r2=0.908836, rmse=4.696488 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 39.1 | 39.1..39.1 | 1 | 0.382 | - | 57.3 | 39.1 | 18.19 (upload_ms_untimed) | 0.261 (whole/whole (kernel not derivable)) | - | 1081.1 | 4649.6 | finite=True, r2=0.908836, rmse=4.696480 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu-eigh | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 7.585 | - | 15.6 | 2.0 | 13.60 (upload_ms_untimed) | 0.960 (whole/whole (kernel not derivable)) | - | 1098.0 | 376.4 | finite=True, r2=0.908836, rmse=4.696490 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0094)`, ran on NVIDIA L40S (nv2, RunPod) job v0094

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 105.3 | 105.3..105.3 | 1 | - | - | 105.3 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0094 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 16.9 | 16.9..16.9 | 1 | 6.247 | - | 91.9 | 16.9 | 75.07 (upload_ms_untimed) | 1.146 (whole/whole (kernel not derivable)) | - | 2997.8 | 3439.6 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 5785dd7f2fd2) | cuml 26.08.00 | opponent | 78.7 | 78.7..78.7 | 1 | 1.338 | - | 1399.8 | 78.7 | 1321.05 (upload_ms_untimed) | 0.075 (whole/whole) | - | 5058.6 | 2190.0 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T19:19:28Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0094)`, ran on NVIDIA L40S (nv2, RunPod) job v0094

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 11.5 | 11.5..11.5 | 1 | - | - | 11.5 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999997 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0094 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 7.989 | - | 9.6 | 1.4 | 8.15 (upload_ms_untimed) | 1.196 (whole/whole (kernel not derivable)) | - | 1058.1 | 344.4 | explained_variance_ratio_sum=0.999996 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 5785dd7f2fd2) | cuml 26.08.00 | opponent | 19.3 | 19.3..19.3 | 1 | 0.594 | - | 191.8 | 19.3 | 172.54 (upload_ms_untimed) | 0.060 (whole/whole) | - | 1309.2 | 642.0 | explained_variance_ratio_sum=0.999996 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T19:19:00Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svc / istella (rows full, shape 10000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0126)`, ran on NVIDIA L40S (nv2, RunPod) job v0126

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 44.3 | 44.3..44.3 | 1 | - | - | 44.3 | - | - (stored whole) | - | - | - | - | accuracy=0.922200, n_support=2400 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0126 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 23.0 | 23.0..23.0 | 1 | 1.927 | - | 39.9 | 23.0 | 16.89 (upload_ms_untimed) | 1.111 (whole/whole) | - | 970.1 | 464.0 | accuracy=0.922200, n_support=2401 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0126)`, ran on NVIDIA L40S (nv2, RunPod) job v0126

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1205.5 | 1205.5..1205.5 | 1 | - | - | 1205.5 | - | - (stored whole) | - | - | - | - | accuracy=0.767500, n_support=5527 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0126 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 377.9 | 377.9..377.9 | 1 | 3.190 | - | 382.4 | 377.9 | 4.54 (upload_ms_untimed) | 3.152 (whole/whole) | - | 1028.3 | 440.0 | accuracy=0.767500, n_support=5541 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical, wave 2

### elasticnet / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 96.9 | 96.9..96.9 | 1 | - | - | 96.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.260922, rmse=0.718134 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 316.2 | 316.2..316.2 | 1 | 0.307 | - | 1116.5 | 316.2 | 800.36 (upload_ms_untimed) | 0.087 (whole/whole) | - | 2909.6 | 1374.0 | finite=True, r2=0.260922, rmse=0.718134 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### elasticnet / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 5.6 | 5.6..5.6 | 1 | - | - | 5.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.907378, rmse=4.847225 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 5.9 | 5.9..5.9 | 1 | 0.961 | - | 65.5 | 5.9 | 59.63 (upload_ms_untimed) | 0.086 (whole/whole) | - | 959.3 | 498.0 | finite=True, r2=0.907378, rmse=4.847224 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on NVIDIA L40S (nv, RunPod) job n0329

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 571.5 | 571.5..571.5 | 1 | - | - | 571.5 | - | - (stored whole) | - | - | - | - | bic=-3.851e+07, mean_log_likelihood=200.794500, n_iter=24 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on NVIDIA L40S (nv, RunPod) job n0329

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 33.1 | 33.1..33.1 | 1 | - | - | 33.1 | - | - (stored whole) | - | - | - | - | bic=-3.668e+06, mean_log_likelihood=12.807640, n_iter=29 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0028)`, ran on NVIDIA L40S (nv2, RunPod) job v0028

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 320.4 | 320.4..320.4 | 1 | - | - | 320.4 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999925, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0028 2026-10-11; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host f003873fb257) | cuvs 26.08.01 | opponent | 429.5 | 429.5..429.5 | 1 | 0.746 | - | 775.8 | 429.5 | 346.37 (upload_ms_untimed) | 0.413 (whole/whole) | - | 1726.9 | 770.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0028)`, ran on NVIDIA L40S (nv2, RunPod) job v0028

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 51.6 | 51.6..51.6 | 1 | - | - | 51.6 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999650, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0028 2026-10-11; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host f003873fb257) | cuvs 26.08.01 | opponent | 219.5 | 219.5..219.5 | 1 | 0.235 | - | 247.2 | 219.5 | 27.75 (upload_ms_untimed) | 0.209 (whole/whole) | - | 931.0 | 450.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1079)`, ran on NVIDIA L40S (nv2, RunPod) job v1079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 105.8 | 105.8..105.8 | 1 | - | - | 105.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310837, rmse=0.693460 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1079 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 953.2 | 953.2..953.2 | 1 | 0.111 | - | 1713.2 | 953.2 | 760.08 (upload_ms_untimed) | 0.062 (whole/whole) | - | 2909.8 | 1374.0 | finite=True, r2=0.310837, rmse=0.693460 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1079)`, ran on NVIDIA L40S (nv2, RunPod) job v1079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 5.9 | 5.9..5.9 | 1 | - | - | 5.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908995, rmse=4.804744 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1079 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 7.1 | 7.1..7.1 | 1 | 0.834 | - | 62.2 | 7.1 | 55.14 (upload_ms_untimed) | 0.095 (whole/whole) | - | 959.1 | 498.0 | finite=True, r2=0.908995, rmse=4.804745 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### logreg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 1797.6 | 1797.6..1797.6 | 1 | - | - | 1797.6 | - | - (stored whole) | - | - | - | - | accuracy=0.924590, logloss=0.181249, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 2136.2 | 2136.2..2136.2 | 1 | 0.842 | - | 2873.5 | 2136.2 | 737.32 (upload_ms_untimed) | 0.626 (whole/whole) | - | 3012.6 | 1374.0 | accuracy=0.924550, logloss=0.181262, nonfinite_proba_rows=0 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### logreg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 20.4 | 20.4..20.4 | 1 | - | - | 20.4 | - | - (stored whole) | - | - | - | - | accuracy=0.763350, logloss=0.538985, nonfinite_proba_rows=0 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 32.1 | 32.1..32.1 | 1 | 0.635 | - | 88.3 | 32.1 | 56.19 (upload_ms_untimed) | 0.231 (whole/whole) | - | 1078.7 | 498.0 | accuracy=0.763340, logloss=0.538984, nonfinite_proba_rows=0 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 526.0 | 526.0..526.0 | 1 | - | - | 526.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328674, rmse=0.684427 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 48.6 | 48.6..48.6 | 1 | 10.828 | - | 806.6 | 48.6 | 758.02 (upload_ms_untimed) | 0.652 (whole/whole) | - | 2961.1 | 1410.0 | finite=True, r2=-0.251259, rmse=0.934403 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0134)`, ran on NVIDIA L40S (nv2, RunPod) job v0134

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 4.6 | 4.6..4.6 | 1 | - | - | 4.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0134 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 9.4 | 9.4..9.4 | 1 | 0.495 | - | 74.4 | 9.4 | 65.09 (upload_ms_untimed) | 0.062 (whole/whole) | - | 990.5 | 534.0 | finite=True, r2=0.908983, rmse=4.805051 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0094)`, ran on NVIDIA L40S (nv2, RunPod) job v0094

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 53.6 | 53.6..53.6 | 1 | - | - | 53.6 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001314 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0094 2026-10-11; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 35.3 | 35.3..35.3 | 1 | 1.519 | - | 726.7 | 35.3 | 691.42 (upload_ms_untimed) | 0.074 (whole/whole) | - | 2738.7 | 1314.0 | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001472 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1226)`, ran on NVIDIA L40S (nv2, RunPod) job v1226

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 2.9 | 2.9..2.9 | 1 | - | - | 2.9 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | main board, one scored run | - | ok (main@a8e0548a5 nv2/v1226 2026-10-10; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 5.9 | 5.9..5.9 | 1 | 0.491 | - | 56.3 | 5.9 | 50.38 (upload_ms_untimed) | 0.052 (whole/whole) | - | 973.4 | 516.0 | explained_variance_ratio_sum=0.999964, relative_reconstruction_error=0.003257 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Neural

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 47.0 ms) = 1.119; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 51.0 ms) = 1.032. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 52.6 | 52.6..52.6 | 1 | - | - | 52.6 | - | - (stored whole) | - | - | - | - | max_rel_err_vs_fp64=2.399e-07 | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 51.0 | 51.0..51.0 | 1 | 1.032 | - | 51.0 | - | - (stored whole) | 1.032 (whole/whole (kernel not derivable)) | - | 934.7 | 200.1 | max_abs_diff_vs_ours=0.0005341, max_rel_diff_vs_ours=1.475e-06, max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 49.4 | 49.4..49.4 | 1 | 1.064 | - | 49.4 | - | - (stored whole) | 1.064 (whole/whole (kernel not derivable)) | - | 936.8 | 200.1 | max_abs_diff_vs_ours=0.100822, max_rel_diff_vs_ours=0.0002785, max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 52.3 | 52.3..52.3 | 1 | 1.006 | - | 52.3 | - | - (stored whole) | 1.006 (whole/whole (kernel not derivable)) | - | 1135.0 | 200.1 | max_abs_diff_vs_ours=0.0005341, max_rel_diff_vs_ours=1.475e-06, max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 50.3 | 50.3..50.3 | 1 | 1.046 | - | 50.3 | - | - (stored whole) | 1.046 (whole/whole (kernel not derivable)) | - | 1136.8 | 200.1 | max_abs_diff_vs_ours=0.100822, max_rel_diff_vs_ours=0.0002785, max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 47.0 | 47.0..47.0 | 1 | 1.119 | - | 47.0 | - | - (stored whole) | 1.119 (whole/whole (kernel not derivable)) | - | 1169.8 | 240.2 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 49.7 | 49.7..49.7 | 1 | 1.059 | - | 49.7 | - | - (stored whole) | 1.059 (whole/whole (kernel not derivable)) | - | 1404.6 | 240.2 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 49.4 ms) = 1.122; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 48.5 ms) = 1.143. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 55.4 | 55.4..55.4 | 1 | - | - | 55.4 | - | - (stored whole) | - | - | - | - | mean_nll=9.018733 | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 49.8 | 49.8..49.8 | 1 | 1.112 | - | 49.8 | - | - (stored whole) | 1.112 (whole/whole (kernel not derivable)) | - | 1061.5 | 173.2 | max_abs_diff_vs_ours=1.371e-06, max_rel_diff_vs_ours=1.882e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 46.5 | 46.5..46.5 | 1 | 1.192 | - | 46.5 | - | - (stored whole) | 1.192 (whole/whole (kernel not derivable)) | - | 1053.9 | 173.2 | max_abs_diff_vs_ours=0.0007233, max_rel_diff_vs_ours=0.0009929, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 48.5 | 48.5..48.5 | 1 | 1.143 | - | 48.5 | - | - (stored whole) | 1.143 (whole/whole (kernel not derivable)) | - | 1172.1 | 154.7 | max_abs_diff_vs_ours=1.341e-06, max_rel_diff_vs_ours=1.841e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 50.0 | 50.0..50.0 | 1 | 1.108 | - | 50.0 | - | - (stored whole) | 1.108 (whole/whole (kernel not derivable)) | - | 1166.8 | 154.7 | max_abs_diff_vs_ours=0.0007184, max_rel_diff_vs_ours=0.0009862, mean_nll=9.018732 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 49.4 | 49.4..49.4 | 1 | 1.122 | - | 49.4 | - | - (stored whole) | 1.122 (whole/whole (kernel not derivable)) | - | 1181.6 | 192.5 | max_abs_diff_vs_ours=0.006870, max_rel_diff_vs_ours=0.009431, mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 51.2 | 51.2..51.2 | 1 | 1.083 | - | 51.2 | - | - (stored whole) | 1.083 (whole/whole (kernel not derivable)) | - | 1329.1 | 192.5 | max_abs_diff_vs_ours=0.006304, max_rel_diff_vs_ours=0.008653, mean_nll=9.018669 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 11.0 ms) = 3.420; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 19.5 ms) = 1.922. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 37.5 | 37.5..37.5 | 1 | - | - | 37.5 | - | - (stored whole) | - | - | - | - | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 22.1 | 22.1..22.1 | 1 | 1.699 | - | 22.1 | - | - (stored whole) | 1.699 (whole/whole (kernel not derivable)) | - | 1261.1 | 959.2 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=9.018733, loss_last_abs_diff_vs_ours=3.815e-06, loss_last_step=8.422415, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 17.7 | 17.7..17.7 | 1 | 2.122 | - | 17.7 | - | - (stored whole) | 2.122 (whole/whole (kernel not derivable)) | - | 1251.7 | 959.2 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018732, loss_last_abs_diff_vs_ours=9.537e-05, loss_last_step=8.422506, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 19.5 | 19.5..19.5 | 1 | 1.922 | - | 19.5 | - | - (stored whole) | 1.922 (whole/whole (kernel not derivable)) | - | 1225.6 | 720.7 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018734, loss_last_abs_diff_vs_ours=3.815e-06, loss_last_step=8.422415, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 20.1 | 20.1..20.1 | 1 | 1.864 | - | 20.1 | - | - (stored whole) | 1.864 (whole/whole (kernel not derivable)) | - | 1222.7 | 727.2 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018732, loss_last_abs_diff_vs_ours=0.0001049, loss_last_step=8.422516, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 18.6 | 18.6..18.6 | 1 | 2.016 | - | 18.6 | - | - (stored whole) | 2.016 (whole/whole (kernel not derivable)) | - | 1387.8 | 796.6 | loss_first_abs_diff_vs_ours=0.0003309, loss_first_step=9.018402, loss_last_abs_diff_vs_ours=0.0008411, loss_last_step=8.421570, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 11.0 | 11.0..11.0 | 1 | 3.420 | - | 11.0 | - | - (stored whole) | 3.420 (whole/whole (kernel not derivable)) | - | 1396.9 | 546.6 | loss_first_abs_diff_vs_ours=6.39e-05, loss_first_step=9.018669, loss_last_abs_diff_vs_ours=0.002161, loss_last_step=8.420250, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0070)`, ran on NVIDIA L40S (nv2, RunPod) job v0070

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 195.9 ms) = 0.028; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 149.6 ms) = 0.036. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 5.4 | 5.4..5.4 | 1 | - | - | 5.4 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0070 2026-10-11; identity vs amd-mi325x: MATCH) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 3.5 | 3.5..3.5 | 1 | 1.541 | - | 3.5 | - | - (stored whole) | 1.541 (whole/whole) | - | 1247.2 | 54.6 | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.945e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.9 | 4.9..4.9 | 1 | 1.105 | - | 4.9 | - | - (stored whole) | 1.105 (whole/whole) | - | 1225.3 | 54.6 | max_abs_diff_vs_ours=3.815e-06, max_rel_diff_vs_ours=1.902e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 195.9 | 195.9..195.9 | 1 | 0.028 | - | 195.9 | - | - (stored whole) | 0.028 (whole/whole (kernel not derivable)) | - | 1189.8 | 307.1 | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.568e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:33:43Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 149.6 | 149.6..149.6 | 1 | 0.036 | - | 149.6 | - | - (stored whole) | 0.036 (whole/whole (kernel not derivable)) | - | 1062.5 | 354.3 | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.945e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:33:43Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 153.5 | 153.5..153.5 | 1 | 0.035 | - | 153.5 | - | - (stored whole) | 0.035 (whole/whole (kernel not derivable)) | - | 1041.7 | 354.3 | max_abs_diff_vs_ours=3.815e-06, max_rel_diff_vs_ours=1.902e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:33:43Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host not sampled; GPU not sampled

memory, mamba-ssm-fp32, mamba-ssm-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 21.7 ms) = 0.361; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 25.8 ms) = 0.304. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 7.8 | 7.8..7.8 | 1 | - | - | 7.8 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 3.9 | 3.9..3.9 | 1 | 1.997 | - | 3.9 | - | - (stored whole) | 1.997 (whole/whole) | - | 1460.8 | 58.6 | max_abs_diff_vs_ours=1.907e-06, max_rel_diff_vs_ours=6.485e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 2.9 | 2.9..2.9 | 1 | 2.692 | - | 2.9 | - | - (stored whole) | 2.692 (whole/whole) | - | 1289.7 | 58.6 | max_abs_diff_vs_ours=0.0005126, max_rel_diff_vs_ours=0.0001743 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 21.7 | 21.7..21.7 | 1 | 0.361 | - | 21.7 | - | - (stored whole) | 0.361 (whole/whole (kernel not derivable)) | - | 1383.4 | 1692.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 26.1 | 26.1..26.1 | 1 | 0.300 | - | 26.1 | - | - (stored whole) | 0.300 (whole/whole (kernel not derivable)) | - | 1250.8 | 3228.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 26.5 | 26.5..26.5 | 1 | 0.296 | - | 26.5 | - | - (stored whole) | 0.296 (whole/whole (kernel not derivable)) | - | 1208.0 | 3228.4 | max_abs_diff_vs_ours=0.000509, max_rel_diff_vs_ours=0.0001731 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 21.7 | 21.7..21.7 | 1 | 0.361 | - | 21.7 | - | - (stored whole) | 0.361 (whole/whole (kernel not derivable)) | - | 1326.0 | 1692.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 25.8 | 25.8..25.8 | 1 | 0.304 | - | 25.8 | - | - (stored whole) | 0.304 (whole/whole (kernel not derivable)) | - | 1192.2 | 3228.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 25.7 | 25.7..25.7 | 1 | 0.305 | - | 25.7 | - | - (stored whole) | 0.305 (whole/whole (kernel not derivable)) | - | 1181.2 | 3228.4 | max_abs_diff_vs_ours=0.000509, max_rel_diff_vs_ours=0.0001731 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:34:20Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host not sampled; GPU not sampled

memory, mamba-ssm-fp32, mamba-ssm-tf32, torch-compile-bf16, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v0999)`, ran on NVIDIA L40S (nv2, RunPod) job v0999

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 9.1 ms) = 0.451; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 13.5 ms) = 0.303. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 4.1 | 4.1..4.1 | 1 | - | - | 4.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@444f0b96e nv2/v0999 2026-10-09; identity vs the amd columns: n/a) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.5 | 4.5..4.5 | 1 | 0.919 | - | 4.5 | - | - (stored whole) | 0.919 (whole/whole) | - | 1414.9 | 70.0 | max_abs_diff_vs_ours=0.001235, max_rel_diff_vs_ours=0.0005401 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.1 | 4.1..4.1 | 1 | 0.999 | - | 4.1 | - | - (stored whole) | 0.999 (whole/whole) | - | 1410.9 | 70.0 | max_abs_diff_vs_ours=0.001376, max_rel_diff_vs_ours=0.0006018 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 9.1 | 9.1..9.1 | 1 | 0.451 | - | 9.1 | - | - (stored whole) | 0.451 (whole/whole (kernel not derivable)) | - | 3809.6 | 90.5 | max_abs_diff_vs_ours=0.001982, max_rel_diff_vs_ours=0.0008672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 13.5 | 13.5..13.5 | 1 | 0.303 | - | 13.5 | - | - (stored whole) | 0.303 (whole/whole (kernel not derivable)) | - | 3465.6 | 83.6 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 9.3 | 9.3..9.3 | 1 | 0.438 | - | 9.3 | - | - (stored whole) | 0.438 (whole/whole (kernel not derivable)) | - | 3068.5 | 83.6 | max_abs_diff_vs_ours=0.0001798, max_rel_diff_vs_ours=7.867e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 83.8 | 83.8..83.8 | 1 | 0.049 | - | 83.8 | - | - (stored whole) | 0.049 (whole/whole (kernel not derivable)) | - | 1313.7 | 212.6 | max_abs_diff_vs_ours=0.002095, max_rel_diff_vs_ours=0.0009164 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 83.7 | 83.7..83.7 | 1 | 0.049 | - | 83.7 | - | - (stored whole) | 0.049 (whole/whole (kernel not derivable)) | - | 1190.9 | 219.7 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 83.9 | 83.9..83.9 | 1 | 0.049 | - | 83.9 | - | - (stored whole) | 0.049 (whole/whole (kernel not derivable)) | - | 1186.6 | 219.7 | max_abs_diff_vs_ours=0.0001827, max_rel_diff_vs_ours=7.993e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T17:51:09Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host not sampled; GPU not sampled

memory, mamba-ssm-fp32, mamba-ssm-tf32, torch-compile-bf16, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.0 ms) = 0.994; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.8 ms) = 1.132. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 2.0 | 2.0..2.0 | 1 | - | - | 2.0 | - | - (stored whole) | - | - | - | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 1.132 | - | 1.8 | - | - (stored whole) | 1.132 (whole/whole (kernel not derivable)) | - | 1013.9 | 16.3 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.207 | - | 1.7 | - | - (stored whole) | 1.207 (whole/whole (kernel not derivable)) | - | 1016.8 | 16.3 | loss_first_abs_diff_vs_ours=8.941e-06, loss_first_step=1.160392, loss_last_abs_diff_vs_ours=5.484e-06, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.866 | - | 2.3 | - | - (stored whole) | 0.866 (whole/whole (kernel not derivable)) | - | 1104.6 | 16.3 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.4 | 2.4..2.4 | 1 | 0.825 | - | 2.4 | - | - (stored whole) | 0.825 (whole/whole (kernel not derivable)) | - | 1048.9 | 16.3 | loss_first_abs_diff_vs_ours=8.941e-06, loss_first_step=1.160392, loss_last_abs_diff_vs_ours=5.603e-06, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.994 | - | 2.0 | - | - (stored whole) | 0.994 (whole/whole (kernel not derivable)) | - | 1228.8 | 16.3 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.2 | 2.2..2.2 | 1 | 0.919 | - | 2.2 | - | - (stored whole) | 0.919 (whole/whole (kernel not derivable)) | - | 1295.5 | 16.3 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001007, loss_last_step=1.123462, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1049)`, ran on NVIDIA L40S (nv2, RunPod) job v1049

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 5.3 ms) = 0.937; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 6.0 ms) = 0.836. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 5.0 | 5.0..5.0 | 1 | - | - | 5.0 | - | - (stored whole) | - | - | - | - | mean_nll=5.635910 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1049 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 46.1 | 46.1..46.1 | 1 | 0.108 | - | 46.1 | - | - (stored whole) | 0.108 (whole/whole (kernel not derivable)) | - | 1260.1 | 137.4 | max_abs_diff_vs_ours=2.235e-06, max_rel_diff_vs_ours=1.257e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 46.4 | 46.4..46.4 | 1 | 0.107 | - | 46.4 | - | - (stored whole) | 0.107 (whole/whole (kernel not derivable)) | - | 1238.9 | 137.4 | max_abs_diff_vs_ours=0.001577, max_rel_diff_vs_ours=0.0008869, mean_nll=5.635948 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 6.0 | 6.0..6.0 | 1 | 0.836 | - | 6.0 | - | - (stored whole) | 0.836 (whole/whole (kernel not derivable)) | - | 1861.2 | 64.2 | max_abs_diff_vs_ours=2.056e-06, max_rel_diff_vs_ours=1.156e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 5.0 | 5.0..5.0 | 1 | 0.993 | - | 5.0 | - | - (stored whole) | 0.993 (whole/whole (kernel not derivable)) | - | 1723.9 | 64.2 | max_abs_diff_vs_ours=0.001510, max_rel_diff_vs_ours=0.0008487, mean_nll=5.635956 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 48.2 | 48.2..48.2 | 1 | 0.104 | - | 48.2 | - | - (stored whole) | 0.104 (whole/whole (kernel not derivable)) | - | 1365.6 | 142.8 | max_abs_diff_vs_ours=0.019621, max_rel_diff_vs_ours=0.011031, mean_nll=5.635952 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 5.3 | 5.3..5.3 | 1 | 0.937 | - | 5.3 | - | - (stored whole) | 0.937 (whole/whole (kernel not derivable)) | - | 2105.1 | 73.6 | max_abs_diff_vs_ours=0.016407, max_rel_diff_vs_ours=0.009224, mean_nll=5.635985 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0067)`, ran on NVIDIA L40S (nv2, RunPod) job v0067

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 157.5 ms) = 0.269; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 57.5 ms) = 0.736. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 42.4 | 42.4..42.4 | 1 | - | - | 42.4 | - | - (stored whole) | - | - | - | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0067 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 57.5 | 57.5..57.5 | 1 | 0.736 | - | 57.5 | - | - (stored whole) | 0.736 (whole/whole (kernel not derivable)) | - | 2932.3 | 311.2 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 43.6 | 43.6..43.6 | 1 | 0.971 | - | 43.6 | - | - (stored whole) | 0.971 (whole/whole (kernel not derivable)) | - | 2371.0 | 316.8 | loss_first_abs_diff_vs_ours=3.815e-05, loss_first_step=5.635948, loss_last_abs_diff_vs_ours=0.0001855, loss_last_step=4.833748, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 157.5 | 157.5..157.5 | 1 | 0.269 | - | 157.5 | - | - (stored whole) | 0.269 (whole/whole (kernel not derivable)) | - | 1519.5 | 330.6 | loss_first_abs_diff_vs_ours=4.196e-05, loss_first_step=5.635952, loss_last_abs_diff_vs_ours=3.338e-05, loss_last_step=4.833967, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 134.2 | 134.2..134.2 | 1 | 0.316 | - | 134.2 | - | - (stored whole) | 0.316 (whole/whole (kernel not derivable)) | - | 1400.9 | 382.4 | loss_first_abs_diff_vs_ours=4.768e-07, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 5785dd7f2fd2) | torch 2.13.0+cu129 | opponent | 147.6 | 147.6..147.6 | 1 | 0.287 | - | 147.6 | - | - (stored whole) | 0.287 (whole/whole (kernel not derivable)) | - | 1394.7 | 382.4 | loss_first_abs_diff_vs_ours=3.815e-05, loss_first_step=5.635948, loss_last_abs_diff_vs_ours=0.0003433, loss_last_step=4.833591, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-grid-logs.txt (nv2/v0139)`, ran on NVIDIA L40S (nv2, RunPod) job v0139

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 3.6 ms) = 2.109; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 3.8 ms) = 2.016. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ff77698) | identical | 7.6 | 7.6..7.6 | 1 | - | - | 7.6 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@c8ff77698 nv2/v0139 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 2.016 | - | 3.8 | - | - (stored whole) | 2.016 (whole/whole (kernel not derivable)) | - | 1097.7 | 78.4 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 1.994 | - | 3.8 | - | - (stored whole) | 1.994 (whole/whole (kernel not derivable)) | - | 1092.7 | 78.4 | max_abs_diff_vs_ours=0.0002124, max_rel_diff_vs_ours=4.545e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.7 | 4.7..4.7 | 1 | 1.620 | - | 4.7 | - | - (stored whole) | 1.620 (whole/whole (kernel not derivable)) | - | 994.8 | 46.9 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 2.199 | - | 3.4 | - | - (stored whole) | 2.199 (whole/whole (kernel not derivable)) | - | 946.1 | 46.9 | max_abs_diff_vs_ours=0.0002124, max_rel_diff_vs_ours=4.545e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.6 | 3.6..3.6 | 1 | 2.109 | - | 3.6 | - | - (stored whole) | 2.109 (whole/whole (kernel not derivable)) | - | 1287.2 | 80.1 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 1.769 | - | 4.3 | - | - (stored whole) | 1.769 (whole/whole (kernel not derivable)) | - | 1228.9 | 41.8 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1065)`, ran on NVIDIA L40S (nv2, RunPod) job v1065

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 15.9 ms) = 0.863. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 13.7 | 13.7..13.7 | 1 | - | - | 13.7 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1065 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 15.9 | 15.9..15.9 | 1 | 0.863 | - | - | 15.9 | - (stored kernel) | 0.863 (MIXED ours whole / arm kernel) | - | 1023.7 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 181.9 | 181.9..181.9 | 1 | 0.075 | - | - | 181.9 | - (stored kernel) | 0.075 (MIXED ours whole / arm kernel) | - | 1105.3 | 1024.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adagrad / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 14.8 ms) = 0.570. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 14.8 | 14.8..14.8 | 1 | 0.570 | - | - | 14.8 | - (stored kernel) | 0.570 (MIXED ours whole / arm kernel) | - | 956.3 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 117.5 | 117.5..117.5 | 1 | 0.072 | - | - | 117.5 | - (stored kernel) | 0.072 (MIXED ours whole / arm kernel) | - | 1282.3 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 18.4 ms) = 0.606. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 11.2 | 11.2..11.2 | 1 | - | - | 11.2 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 18.4 | 18.4..18.4 | 1 | 0.606 | - | - | 18.4 | - (stored kernel) | 0.606 (MIXED ours whole / arm kernel) | - | 969.4 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 155.4 | 155.4..155.4 | 1 | 0.072 | - | - | 155.4 | - (stored kernel) | 0.072 (MIXED ours whole / arm kernel) | - | 1428.8 | 896.0 | rel_fro_vs_torch_eager_fp32=6.395e-09 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ard / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0079)`, ran on NVIDIA L40S (nv2, RunPod) job v0079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 317.7 | 317.7..317.7 | 1 | - | - | 317.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.124856, rmse=0.885949 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0079 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 7400.8 | 7400.8..7400.8 | 1 | 0.043 | - | 7400.8 | 7400.8 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 1381.8 | - | finite=True, r2=0.327402, rmse=0.685075 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### autoarima / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0031)`, ran on NVIDIA L40S (nv2, RunPod) job v0031

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 12843.5 | 12843.5..12843.5 | 1 | - | - | 12843.5 | - | - (stored whole) | - | - | - | - | forecast_rmse=3.260829 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0031 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 7452.1 | 7452.1..7452.1 | 1 | 1.723 | - | 7452.1 | 7452.1 | 0.00 (cpu-arm) | 1.723 (whole/whole) | - | 467.2 | - | forecast_rmse=17.543262 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

memory, ours, cuml-gpu: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### autoarima / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0031)`, ran on NVIDIA L40S (nv2, RunPod) job v0031

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 14474.8 | 14474.8..14474.8 | 1 | - | - | 14474.8 | - | - (stored whole) | - | - | - | - | forecast_rmse=73.388358 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0031 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 6148.5 | 6148.5..6148.5 | 1 | 2.354 | - | 6148.5 | 6148.5 | 0.00 (cpu-arm) | 2.354 (whole/whole) | - | 466.7 | - | forecast_rmse=68.213954 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 157606.5 | 157606.5..157606.5 | 1 | 0.092 | - | 157610.9 | 157606.5 | 4.32 (upload_ms_untimed) | 0.092 (whole/whole) | - | 835.9 | 812.0 | forecast_rmse=106.817545 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.1 ms) = 0.885; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 0.9 ms) = 1.114. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.0 | 1.0..1.0 | 1 | - | - | 1.0 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 0.883 | - | - | 1.1 | - (stored kernel) | 0.883 (MIXED ours whole / arm kernel) | - | 673.3 | 224.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 0.9 | 0.9..0.9 | 1 | 1.114 | - | - | 0.9 | - (stored kernel) | 1.114 (MIXED ours whole / arm kernel) | - | 912.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 0.896 | - | - | 1.1 | - (stored kernel) | 0.896 (MIXED ours whole / arm kernel) | - | 683.3 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.653 | - | - | 1.5 | - (stored kernel) | 0.653 (MIXED ours whole / arm kernel) | - | 862.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 0.885 | - | - | 1.1 | - (stored kernel) | 0.885 (MIXED ours whole / arm kernel) | - | 673.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.535 | - | - | 1.8 | - (stored kernel) | 0.535 (MIXED ours whole / arm kernel) | - | 857.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0082)`, ran on NVIDIA L40S (nv2, RunPod) job v0082

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.6 ms) = 2.898; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.4 ms) = 3.182. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 4.5 | 4.5..4.5 | 1 | - | - | 4.5 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0082 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 2.997 | - | - | 1.5 | - (stored kernel) | 2.997 (MIXED ours whole / arm kernel) | - | 699.1 | 541.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 3.182 | - | - | 1.4 | - (stored kernel) | 3.182 (MIXED ours whole / arm kernel) | - | 929.3 | 540.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 2.931 | - | - | 1.5 | - (stored kernel) | 2.931 (MIXED ours whole / arm kernel) | - | 699.3 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 2.753 | - | - | 1.6 | - (stored kernel) | 2.753 (MIXED ours whole / arm kernel) | - | 880.4 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 2.898 | - | - | 1.6 | - (stored kernel) | 2.898 (MIXED ours whole / arm kernel) | - | 698.2 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 2.733 | - | - | 1.7 | - (stored kernel) | 2.733 (MIXED ours whole / arm kernel) | - | 873.9 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0140)`, ran on NVIDIA L40S (nv2, RunPod) job v0140

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.6 ms) = 0.989; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.8 ms) = 0.889. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@7924718fc) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@7924718fc nv2/v0140 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.883 | - | - | 1.8 | - (stored kernel) | 0.883 (MIXED ours whole / arm kernel) | - | 762.5 | 320.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.889 | - | - | 1.8 | - (stored kernel) | 0.889 (MIXED ours whole / arm kernel) | - | 969.0 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 1.014 | - | - | 1.5 | - (stored kernel) | 1.014 (MIXED ours whole / arm kernel) | - | 763.7 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.796 | - | - | 2.0 | - (stored kernel) | 0.796 (MIXED ours whole / arm kernel) | - | 905.0 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.973 | - | - | 1.6 | - (stored kernel) | 0.973 (MIXED ours whole / arm kernel) | - | 762.6 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.989 | - | - | 1.6 | - (stored kernel) | 0.989 (MIXED ours whole / arm kernel) | - | 897.3 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0140)`, ran on NVIDIA L40S (nv2, RunPod) job v0140

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 0.836; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.4 ms) = 0.879. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@7924718fc) | identical | 1.3 | 1.3..1.3 | 1 | - | - | 1.3 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@7924718fc nv2/v0140 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.862 | - | - | 1.5 | - (stored kernel) | 0.862 (MIXED ours whole / arm kernel) | - | 748.9 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 0.879 | - | - | 1.4 | - (stored kernel) | 0.879 (MIXED ours whole / arm kernel) | - | 952.6 | 246.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 0.902 | - | - | 1.4 | - (stored kernel) | 0.902 (MIXED ours whole / arm kernel) | - | 748.9 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.781 | - | - | 1.6 | - (stored kernel) | 0.781 (MIXED ours whole / arm kernel) | - | 890.4 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.836 | - | - | 1.5 | - (stored kernel) | 0.836 (MIXED ours whole / arm kernel) | - | 749.1 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.692 | - | - | 1.8 | - (stored kernel) | 0.692 (MIXED ours whole / arm kernel) | - | 882.1 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bayesian-ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0079)`, ran on NVIDIA L40S (nv2, RunPod) job v0079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 611.4 | 611.4..611.4 | 1 | - | - | 611.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.317909, rmse=0.689893 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0079 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 8161.0 | 8161.0..8161.0 | 1 | 0.075 | - | 8161.0 | 8161.0 | 0.00 (cpu-arm) | 0.075 (whole/whole) | - | 3653.3 | - | finite=True, r2=-395441.818968, rmse=525.293799 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 52.3 | 52.3..52.3 | 1 | - | - | 52.3 | - | - (stored whole) | - | - | - | - | accuracy=0.794050, logloss=5.350625 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1027.1 | 1027.1..1027.1 | 1 | 0.051 | - | 1027.1 | 1027.1 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 3823.4 | - | accuracy=0.794050, logloss=4.278741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 140.3 | 140.3..140.3 | 1 | 0.373 | - | 914.9 | 140.3 | 774.63 (upload_ms_untimed) | 0.057 (whole/whole) | - | 2960.1 | 1370.0 | accuracy=0.794050, logloss=5.350631 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 13.5 | 13.5..13.5 | 1 | - | - | 13.5 | - | - (stored whole) | - | - | - | - | accuracy=0.755560, logloss=0.557803 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 121.5 | 121.5..121.5 | 1 | 0.111 | - | 121.5 | 121.5 | 0.00 (cpu-arm) | 0.111 (whole/whole) | - | 431.8 | - | accuracy=0.755560, logloss=0.557802 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 13.1 | 13.1..13.1 | 1 | 1.031 | - | 72.3 | 13.1 | 59.16 (upload_ms_untimed) | 0.187 (whole/whole) | - | 1123.1 | 494.0 | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 38.0 | 38.0..38.0 | 1 | 0.004 | - | 38.0 | 38.0 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1499.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 4.4 | 4.4..4.4 | 1 | 0.034 | - | 768.1 | 4.4 | 763.63 (upload_ms_untimed) | 0.0001937 (whole/whole) | - | 2955.5 | 1440.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2.2 | 2.2..2.2 | 1 | 0.058 | - | 2.2 | 2.2 | 0.00 (cpu-arm) | 0.058 (whole/whole) | - | 267.4 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 0.9 | 0.9..0.9 | 1 | 0.135 | - | 62.0 | 0.9 | 61.11 (upload_ms_untimed) | 0.002 (whole/whole) | - | 869.4 | 486.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 21.0 | 21.0..21.0 | 1 | - | - | 21.0 | - | - (stored whole) | - | - | - | - | accuracy=0.838850, logloss=0.412625 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 137.8 | 137.8..137.8 | 1 | 0.152 | - | 137.8 | 137.8 | 0.00 (cpu-arm) | 0.152 (whole/whole) | - | 352.7 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 17.6 | 17.6..17.6 | 1 | 1.192 | - | 47.6 | 17.6 | 30.05 (upload_ms_untimed) | 0.440 (whole/whole) | - | 908.6 | 476.0 | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 17.1 | 17.1..17.1 | 1 | - | - | 17.1 | - | - (stored whole) | - | - | - | - | accuracy=0.765850, logloss=0.538866 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 117.0 | 117.0..117.0 | 1 | 0.146 | - | 117.0 | 117.0 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 315.0 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 14.1 | 14.1..14.1 | 1 | 1.208 | - | 48.5 | 14.1 | 34.39 (upload_ms_untimed) | 0.352 (whole/whole) | - | 1086.4 | 460.0 | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on scikit-learn; ours refuses the option (option parity) and cuML has none

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0001)`, ran on NVIDIA L40S (nv2, RunPod) job v0001

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@5876e9205) | identical | 212.8 | 212.8..212.8 | 1 | - | - | 212.8 | - | - (stored whole) | - | - | - | - | relative_residual=2.9e-07 | - | main board, one scored run | - | ok (main@5876e9205 nv2/v0001 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 4494.5 | 4494.5..4494.5 | 1 | 0.047 | - | 4494.5 | 4494.5 | 0.00 (cpu-arm) | 0.047 (whole/whole) | - | 2677.2 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 12.7 | 12.7..12.7 | 1 | 16.774 | - | - | 12.7 | - (stored kernel) | 16.774 (MIXED ours whole / arm kernel) | - | 1191.6 | 768.3 | relative_residual=1.509e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 15.2 | 15.2..15.2 | 1 | 13.965 | - | - | 15.2 | - (stored kernel) | 13.965 (MIXED ours whole / arm kernel) | - | 1274.7 | 1234.0 | relative_residual=1.365e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0065)`, ran on NVIDIA L40S (nv2, RunPod) job v0065

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 302.4 ms) = 0.663; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 208.8 ms) = 0.959. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 200.4 | 200.4..200.4 | 1 | - | - | 200.4 | - | - (stored whole) | - | - | - | - | accuracy=1.000000 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0065 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 208.8 | 208.8..208.8 | 1 | 0.959 | - | - | 208.8 | - (stored kernel) | 0.959 (MIXED ours whole / arm kernel) | - | 1297.9 | 287.8 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 265.1 | 265.1..265.1 | 1 | 0.756 | - | - | 265.1 | - (stored kernel) | 0.756 (MIXED ours whole / arm kernel) | - | 1510.4 | 214.3 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 219.4 | 219.4..219.4 | 1 | 0.913 | - | - | 219.4 | - (stored kernel) | 0.913 (MIXED ours whole / arm kernel) | - | 1298.8 | 287.8 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 274.9 | 274.9..274.9 | 1 | 0.729 | - | - | 274.9 | - (stored kernel) | 0.729 (MIXED ours whole / arm kernel) | - | 1337.5 | 214.3 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 302.4 | 302.4..302.4 | 1 | 0.663 | - | - | 302.4 | - (stored kernel) | 0.663 (MIXED ours whole / arm kernel) | - | 1530.8 | 204.7 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 366.5 | 366.5..366.5 | 1 | 0.547 | - | - | 366.5 | - (stored kernel) | 0.547 (MIXED ours whole / arm kernel) | - | 1598.0 | 154.0 | accuracy=1.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 116.0 | 116.0..116.0 | 1 | - | - | 116.0 | - | - (stored whole) | - | - | - | - | accuracy=0.849360, logloss=3.762524 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 278.9 | 278.9..278.9 | 1 | 0.416 | - | 278.9 | 278.9 | 0.00 (cpu-arm) | 0.416 (whole/whole) | - | 3912.0 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 60.2 | 60.2..60.2 | 1 | 1.927 | - | 899.0 | 60.2 | 838.77 (upload_ms_untimed) | 0.129 (whole/whole) | - | 3888.1 | 1372.0 | accuracy=0.849350, logloss=3.763060 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 29.3 | 29.3..29.3 | 1 | - | - | 29.3 | - | - (stored whole) | - | - | - | - | accuracy=0.678020, logloss=0.715492 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 70.7 | 70.7..70.7 | 1 | 0.415 | - | 70.7 | 70.7 | 0.00 (cpu-arm) | 0.415 (whole/whole) | - | 434.5 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 12.5 | 12.5..12.5 | 1 | 2.356 | - | 78.4 | 12.5 | 65.94 (upload_ms_untimed) | 0.374 (whole/whole) | - | 1054.3 | 496.0 | accuracy=0.678060, logloss=0.715531 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on NVIDIA L40S (nv, RunPod) job n0532

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 161.9 | 161.9..161.9 | 1 | - | - | 161.9 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559491 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 410.6 | 410.6..410.6 | 1 | 0.394 | - | 410.6 | 410.6 | 0.00 (cpu-arm) | 0.394 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 119.6 | 119.6..119.6 | 1 | 1.354 | - | 1599.5 | 119.6 | 1479.90 (upload_ms_untimed) | 0.101 (whole/whole) | - | 4711.8 | 1946.0 | accuracy=0.983067, logloss=0.559490 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 0.8 | 0.8..0.8 | 1 | - | - | 0.8 | - | - (stored whole) | - | - | - | - | n_components=81 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 13.2 | 13.2..13.2 | 1 | 0.062 | - | 13.2 | 13.2 | 0.00 (cpu-arm) | 0.062 (whole/whole) | - | 117.1 | - | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host f003873fb257) | cugraph 26.08.00 | opponent | 18.8 | 18.8..18.8 | 1 | 0.044 | - | 18.8 | 18.8 | 0.01 (upload_ms_untimed) | 0.044 (whole/whole) | - | 922.6 | 432.0 | ari_vs_networkx=1.000000, n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | n_components=588 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 18.7 | 18.7..18.7 | 1 | 0.086 | - | 18.7 | 18.7 | 0.00 (cpu-arm) | 0.086 (whole/whole) | - | 99.0 | - | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host f003873fb257) | cugraph 26.08.00 | opponent | 31.2 | 31.2..31.2 | 1 | 0.051 | - | 31.2 | 31.2 | 0.01 (upload_ms_untimed) | 0.051 (whole/whole) | - | 1032.3 | 432.0 | ari_vs_networkx=1.000000, n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0082)`, ran on NVIDIA L40S (nv2, RunPod) job v0082

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 3.3 ms) = 3.085; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 3.9 ms) = 2.585. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 10.0 | 10.0..10.0 | 1 | - | - | 10.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v0082/work-conv1d-synthetic-def/conv1d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0082 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.9 | 3.9..3.9 | 1 | 2.585 | - | - | 3.9 | - (stored kernel) | 2.585 (MIXED ours whole / arm kernel) | - | 971.3 | 640.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.4 | 4.4..4.4 | 1 | 2.305 | - | - | 4.4 | - (stored kernel) | 2.305 (MIXED ours whole / arm kernel) | - | 1163.7 | 640.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.7 | 3.7..3.7 | 1 | 2.728 | - | - | 3.7 | - (stored kernel) | 2.728 (MIXED ours whole / arm kernel) | - | 987.8 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.9 | 3.9..3.9 | 1 | 2.546 | - | - | 3.9 | - (stored kernel) | 2.546 (MIXED ours whole / arm kernel) | - | 1131.2 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.3 | 3.3..3.3 | 1 | 3.085 | - | - | 3.3 | - (stored kernel) | 3.085 (MIXED ours whole / arm kernel) | - | 962.0 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3417.849541, rel_fro_vs_torch_eager_fp32=0.003296 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.3 | 3.3..3.3 | 1 | 3.001 | - | - | 3.3 | - (stored kernel) | 3.001 (MIXED ours whole / arm kernel) | - | 1163.4 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3524.661064, rel_fro_vs_torch_eager_fp32=0.003294 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0082)`, ran on NVIDIA L40S (nv2, RunPod) job v0082

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 4.073; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.3 ms) = 2.774. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 6.3 | 6.3..6.3 | 1 | - | - | 6.3 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v0082/work-conv2d-synthetic-def/conv2d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0082 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 2.774 | - | - | 2.3 | - (stored kernel) | 2.774 (MIXED ours whole / arm kernel) | - | 894.4 | 250.5 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.6 | 2.6..2.6 | 1 | 2.413 | - | - | 2.6 | - (stored kernel) | 2.413 (MIXED ours whole / arm kernel) | - | 1114.4 | 346.8 | max_rel_diff_vs_torch_eager_fp32=0.506768, rel_fro_vs_torch_eager_fp32=5.007e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 3.576 | - | - | 1.8 | - (stored kernel) | 3.576 (MIXED ours whole / arm kernel) | - | 911.7 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.2 | 2.2..2.2 | 1 | 2.839 | - | - | 2.2 | - (stored kernel) | 2.839 (MIXED ours whole / arm kernel) | - | 1065.1 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 4.073 | - | - | 1.5 | - (stored kernel) | 4.073 (MIXED ours whole / arm kernel) | - | 949.9 | 273.7 | max_rel_diff_vs_torch_eager_fp32=3418.337554, rel_fro_vs_torch_eager_fp32=0.003382 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 3.336 | - | - | 1.9 | - (stored kernel) | 3.336 (MIXED ours whole / arm kernel) | - | 1158.8 | 272.7 | max_rel_diff_vs_torch_eager_fp32=3433.596343, rel_fro_vs_torch_eager_fp32=0.003380 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 675.3 | 675.3..675.3 | 1 | - | - | 675.3 | - | - (stored whole) | - | - | - | - | forecast_rmse=13.931153 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 173.6 | 173.6..173.6 | 1 | 3.889 | - | 173.6 | 173.6 | 0.00 (cpu-arm) | 3.889 (whole/whole) | - | 58.6 | - | forecast_rmse=26.588738 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 164.3 | 164.3..164.3 | 1 | 4.109 | - | 164.3 | 164.3 | 0.00 (cpu-arm) | 4.109 (whole/whole) | - | 468.0 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 743.2 | 743.2..743.2 | 1 | - | - | 743.2 | - | - (stored whole) | - | - | - | - | forecast_rmse=96.690449 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 173.0 | 173.0..173.0 | 1 | 4.297 | - | 173.0 | 173.0 | 0.00 (cpu-arm) | 4.297 (whole/whole) | - | 58.2 | - | forecast_rmse=196.955273 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 133.9 | 133.9..133.9 | 1 | 5.551 | - | 133.9 | 133.9 | 0.00 (cpu-arm) | 5.551 (whole/whole) | - | 468.2 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.4 ms) = 0.772. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 5.9 | 5.9..5.9 | 1 | 0.188 | - | - | 5.9 | - (stored kernel) | 0.188 (MIXED ours whole / arm kernel) | - | 683.5 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 0.772 | - | - | 1.4 | - (stored kernel) | 0.772 (MIXED ours whole / arm kernel) | - | 902.2 | 247.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.0 | 1.0..1.0 | 1 | 1.164 | - | - | 1.0 | - (stored kernel) | 1.164 (MIXED ours whole / arm kernel) | - | 683.2 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.581 | - | - | 1.9 | - (stored kernel) | 0.581 (MIXED ours whole / arm kernel) | - | 838.9 | 250.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dynamic-optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 337.4 | 337.4..337.4 | 1 | - | - | 337.4 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436278 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1030.6 | 1030.6..1030.6 | 1 | 0.327 | - | 1030.6 | 1030.6 | 0.00 (cpu-arm) | 0.327 (whole/whole) | - | 467.8 | - | forecast_rmse=1.436045 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dynamic-optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1789.9 | 1789.9..1789.9 | 1 | - | - | 1789.9 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.086628 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1829.8 | 1829.8..1829.8 | 1 | 0.978 | - | 1829.8 | 1829.8 | 0.00 (cpu-arm) | 0.978 (whole/whole) | - | 467.5 | - | forecast_rmse=49.314797 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0051)`, ran on NVIDIA L40S (nv2, RunPod) job v0051

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 403.3 | 403.3..403.3 | 1 | - | - | 403.3 | - | - (stored whole) | - | - | - | - | max_eigenvalue_error=3.692e-07, relative_residual=6.672e-07 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0051 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 14605.9 | 14605.9..14605.9 | 1 | 0.028 | - | 14605.9 | 14605.9 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 945.8 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 80.0 | 80.0..80.0 | 1 | 5.044 | - | - | 80.0 | - (stored kernel) | 5.044 (MIXED ours whole / arm kernel) | - | 913.4 | 450.1 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 80.1 | 80.1..80.1 | 1 | 5.037 | - | - | 80.1 | - (stored kernel) | 5.037 (MIXED ours whole / arm kernel) | - | 710.3 | 1114.0 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### embedding / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 2.1 ms) = 1.290. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 2.7 | 2.7..2.7 | 1 | - | - | 2.7 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 1.096 | - | - | 2.5 | - (stored kernel) | 1.096 (MIXED ours whole / arm kernel) | - | 832.0 | 782.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.1 | 2.1..2.1 | 1 | 1.290 | - | - | 2.1 | - (stored kernel) | 1.290 (MIXED ours whole / arm kernel) | - | 1159.8 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 464.6 | 464.6..464.6 | 1 | - | - | 464.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.326805, rmse=0.685379 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 112446.5 | 112446.5..112446.5 | 1 | 0.004 | - | 112446.5 | 112446.5 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 8909.5 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 19.4 | 19.4..19.4 | 1 | - | - | 19.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909004, rmse=4.804486 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 507.5 | 507.5..507.5 | 1 | 0.038 | - | 507.5 | 507.5 | 0.00 (cpu-arm) | 0.038 (whole/whole) | - | 645.0 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0668)`, ran on NVIDIA L40S (nv, RunPod) job n0668

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@6b882cf81) | identical | 92.1 | 92.1..92.1 | 1 | - | - | 92.1 | - | - (stored whole) | - | - | - | - | accuracy=0.876570, logloss=3.574420 | - | main board, one scored run | - | ok (main@6b882cf81 nv/n0668 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 595.4 | 595.4..595.4 | 1 | 0.155 | - | 595.4 | 595.4 | 0.00 (cpu-arm) | 0.155 (whole/whole) | - | 2625.7 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 127.1 | 127.1..127.1 | 1 | 0.725 | - | 886.9 | 127.1 | 759.83 (upload_ms_untimed) | 0.104 (whole/whole) | - | 2878.3 | 1362.0 | accuracy=0.876570, logloss=3.416741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0668)`, ran on NVIDIA L40S (nv, RunPod) job n0668

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@6b882cf81) | identical | 20.0 | 20.0..20.0 | 1 | - | - | 20.0 | - | - (stored whole) | - | - | - | - | accuracy=0.719820, logloss=1.132249 | - | main board, one scored run | - | ok (main@6b882cf81 nv/n0668 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 137.0 | 137.0..137.0 | 1 | 0.146 | - | 137.0 | 137.0 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 314.3 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 22.5 | 22.5..22.5 | 1 | 0.887 | - | 80.7 | 22.5 | 58.15 (upload_ms_untimed) | 0.248 (whole/whole) | - | 957.7 | 486.0 | accuracy=0.719810, logloss=1.132317 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 54.3 | 54.3..54.3 | 1 | - | - | 54.3 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.680693 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 32.3 | 32.3..32.3 | 1 | 1.684 | - | 32.3 | 32.3 | 0.00 (cpu-arm) | 1.684 (whole/whole) | - | 1134.6 | - | mean_abs_distortion=0.177966 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 2.4 | 2.4..2.4 | 1 | 22.244 | - | 745.9 | 2.4 | 743.42 (upload_ms_untimed) | 0.073 (whole/whole) | - | 2784.6 | 1282.0 | mean_abs_distortion=0.443920 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.345752 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.6 | 3.6..3.6 | 1 | 0.957 | - | 3.6 | 3.6 | 0.00 (cpu-arm) | 0.957 (whole/whole) | - | 257.1 | - | mean_abs_distortion=0.339791 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1.1 | 1.1..1.1 | 1 | 3.120 | - | 59.9 | 1.1 | 58.77 (upload_ms_untimed) | 0.057 (whole/whole) | - | 907.5 | 486.0 | mean_abs_distortion=0.302259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 5.3 ms) = 0.571; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.8 ms) = 0.804. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-gcn-istella-def/gcn-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 13.9 | 13.9..13.9 | 1 | 0.217 | - | - | 13.9 | - (stored kernel) | 0.217 (MIXED ours whole / arm kernel) | - | 1412.6 | 1934.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 0.804 | - | - | 3.8 | - (stored kernel) | 0.804 (MIXED ours whole / arm kernel) | - | 1325.8 | 399.4 | max_rel_diff_vs_torch_eager_fp32=0.016245, rel_fro_vs_torch_eager_fp32=1.041e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 13.7 | 13.7..13.7 | 1 | 0.220 | - | - | 13.7 | - (stored kernel) | 0.220 (MIXED ours whole / arm kernel) | - | 1404.5 | 1934.6 | max_rel_diff_vs_torch_eager_fp32=281.178644, rel_fro_vs_torch_eager_fp32=0.0002661 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.9 | 3.9..3.9 | 1 | 0.766 | - | - | 3.9 | - (stored kernel) | 0.766 (MIXED ours whole / arm kernel) | - | 1279.1 | 399.4 | max_rel_diff_vs_torch_eager_fp32=281.174567, rel_fro_vs_torch_eager_fp32=0.0002661 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 13.9 | 13.9..13.9 | 1 | 0.217 | - | - | 13.9 | - (stored kernel) | 0.217 (MIXED ours whole / arm kernel) | - | 1528.8 | 2323.8 | max_rel_diff_vs_torch_eager_fp32=2718.608563, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 5.3 | 5.3..5.3 | 1 | 0.571 | - | - | 5.3 | - (stored kernel) | 0.571 (MIXED ours whole / arm kernel) | - | 1560.4 | 782.4 | max_rel_diff_vs_torch_eager_fp32=2718.611825, rel_fro_vs_torch_eager_fp32=0.002187 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 8.8 ms) = 0.336; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.2 ms) = 0.927. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-gcn-taxi-def/gcn-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 11.0 | 11.0..11.0 | 1 | 0.268 | - | - | 11.0 | - (stored kernel) | 0.268 (MIXED ours whole / arm kernel) | - | 1328.6 | 1589.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.2 | 3.2..3.2 | 1 | 0.927 | - | - | 3.2 | - (stored kernel) | 0.927 (MIXED ours whole / arm kernel) | - | 1266.4 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002328, rel_fro_vs_torch_eager_fp32=9.806e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 11.1 | 11.1..11.1 | 1 | 0.267 | - | - | 11.1 | - (stored kernel) | 0.267 (MIXED ours whole / arm kernel) | - | 1328.3 | 1589.7 | max_rel_diff_vs_torch_eager_fp32=0.001397, rel_fro_vs_torch_eager_fp32=7.217e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 0.880 | - | - | 3.4 | - (stored kernel) | 0.880 (MIXED ours whole / arm kernel) | - | 1228.2 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002215, rel_fro_vs_torch_eager_fp32=9.792e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 16.0 | 16.0..16.0 | 1 | 0.185 | - | - | 16.0 | - (stored kernel) | 0.185 (MIXED ours whole / arm kernel) | - | 1459.6 | 1874.9 | max_rel_diff_vs_torch_eager_fp32=1509.509282, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 8.8 | 8.8..8.8 | 1 | 0.336 | - | - | 8.8 | - (stored kernel) | 0.336 (MIXED ours whole / arm kernel) | - | 1448.3 | 591.0 | max_rel_diff_vs_torch_eager_fp32=1509.509049, rel_fro_vs_torch_eager_fp32=0.002330 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0082)`, ran on NVIDIA L40S (nv2, RunPod) job v0082

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.9 ms) = 2.733; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.6 ms) = 4.212. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 2.4 | 2.4..2.4 | 1 | - | - | 2.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v0082/work-global-avgpool-synthetic-def/global-avgpool-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0082 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 4.212 | - | - | 0.6 | - (stored kernel) | 4.212 (MIXED ours whole / arm kernel) | - | 680.4 | 12.6 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.0 | 1.0..1.0 | 1 | 2.328 | - | - | 1.0 | - (stored kernel) | 2.328 (MIXED ours whole / arm kernel) | - | 881.9 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 0.7 | 0.7..0.7 | 1 | 3.249 | - | - | 0.7 | - (stored kernel) | 3.249 (MIXED ours whole / arm kernel) | - | 680.4 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.6 | 2.6..2.6 | 1 | 0.922 | - | - | 2.6 | - (stored kernel) | 0.922 (MIXED ours whole / arm kernel) | - | 830.8 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 0.9 | 0.9..0.9 | 1 | 2.733 | - | - | 0.9 | - (stored kernel) | 2.733 (MIXED ours whole / arm kernel) | - | 680.4 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 2.138 | - | - | 1.1 | - (stored kernel) | 2.138 (MIXED ours whole / arm kernel) | - | 826.1 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0082)`, ran on NVIDIA L40S (nv2, RunPod) job v0082

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.2 ms) = 2.648; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.6 ms) = 5.409. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0082 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 5.409 | - | - | 0.6 | - (stored kernel) | 5.409 (MIXED ours whole / arm kernel) | - | 657.3 | 12.9 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 3.024 | - | - | 1.1 | - (stored kernel) | 3.024 (MIXED ours whole / arm kernel) | - | 892.5 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 0.5 | 0.5..0.5 | 1 | 6.789 | - | - | 0.5 | - (stored kernel) | 6.789 (MIXED ours whole / arm kernel) | - | 656.6 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 1.834 | - | - | 1.8 | - (stored kernel) | 1.834 (MIXED ours whole / arm kernel) | - | 842.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 2.034 | - | - | 1.6 | - (stored kernel) | 2.034 (MIXED ours whole / arm kernel) | - | 657.3 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 2.648 | - | - | 1.2 | - (stored kernel) | 2.648 (MIXED ours whole / arm kernel) | - | 839.6 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.5 ms) = 2.159; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.9 ms) = 1.925. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 7.5 | 7.5..7.5 | 1 | - | - | 7.5 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-graphsage-istella-def/graphsage-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 8.5 | 8.5..8.5 | 1 | 0.875 | - | - | 8.5 | - (stored kernel) | 0.875 (MIXED ours whole / arm kernel) | - | 1246.6 | 1667.4 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.9 | 3.9..3.9 | 1 | 1.925 | - | - | 3.9 | - (stored kernel) | 1.925 (MIXED ours whole / arm kernel) | - | 1292.9 | 403.7 | max_rel_diff_vs_torch_eager_fp32=0.119209, rel_fro_vs_torch_eager_fp32=9.699e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 8.4 | 8.4..8.4 | 1 | 0.892 | - | - | 8.4 | - (stored kernel) | 0.892 (MIXED ours whole / arm kernel) | - | 1241.4 | 1667.4 | max_rel_diff_vs_torch_eager_fp32=430.904329, rel_fro_vs_torch_eager_fp32=0.0002969 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.7 | 3.7..3.7 | 1 | 1.999 | - | - | 3.7 | - (stored kernel) | 1.999 (MIXED ours whole / arm kernel) | - | 1239.7 | 403.8 | max_rel_diff_vs_torch_eager_fp32=430.904329, rel_fro_vs_torch_eager_fp32=0.0002969 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 8.1 | 8.1..8.1 | 1 | 0.918 | - | - | 8.1 | - (stored kernel) | 0.918 (MIXED ours whole / arm kernel) | - | 1382.3 | 1642.9 | max_rel_diff_vs_torch_eager_fp32=3907.084465, rel_fro_vs_torch_eager_fp32=0.003366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.5 | 3.5..3.5 | 1 | 2.159 | - | - | 3.5 | - (stored kernel) | 2.159 (MIXED ours whole / arm kernel) | - | 1424.6 | 330.6 | max_rel_diff_vs_torch_eager_fp32=3678.202629, rel_fro_vs_torch_eager_fp32=0.003054 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.8 ms) = 0.978; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.4 ms) = 1.253. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.7 | 1.7..1.7 | 1 | - | - | 1.7 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-graphsage-taxi-def/graphsage-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 1.253 | - | - | 1.4 | - (stored kernel) | 1.253 (MIXED ours whole / arm kernel) | - | 1163.2 | 289.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.6 | 2.6..2.6 | 1 | 0.662 | - | - | 2.6 | - (stored kernel) | 0.662 (MIXED ours whole / arm kernel) | - | 1229.0 | 240.0 | max_rel_diff_vs_torch_eager_fp32=0.238419, rel_fro_vs_torch_eager_fp32=6.978e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.007 | - | - | 1.7 | - (stored kernel) | 1.007 (MIXED ours whole / arm kernel) | - | 1164.7 | 289.1 | max_rel_diff_vs_torch_eager_fp32=0.083333, rel_fro_vs_torch_eager_fp32=4.164e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.928 | - | - | 1.9 | - (stored kernel) | 0.928 (MIXED ours whole / arm kernel) | - | 1223.0 | 240.0 | max_rel_diff_vs_torch_eager_fp32=1475.334167, rel_fro_vs_torch_eager_fp32=0.0002145 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 3.1 | 3.1..3.1 | 1 | 0.557 | - | - | 3.1 | - (stored kernel) | 0.557 (MIXED ours whole / arm kernel) | - | 1298.5 | 262.6 | max_rel_diff_vs_torch_eager_fp32=5460.333333, rel_fro_vs_torch_eager_fp32=0.003557 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.978 | - | - | 1.8 | - (stored kernel) | 0.978 (MIXED ours whole / arm kernel) | - | 1346.6 | 167.2 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 717.1 ms) = 0.623; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 555.9 ms) = 0.804. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.1 | 447.1..447.1 | 1 | - | - | 447.1 | - | - (stored whole) | - | - | - | - | accuracy=0.971625, logloss=0.070054 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 862.4 | 862.4..862.4 | 1 | 0.518 | - | - | 862.4 | - (stored kernel) | 0.518 (MIXED ours whole / arm kernel) | - | 1098.6 | 643.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 555.9 | 555.9..555.9 | 1 | 0.804 | - | - | 555.9 | - (stored kernel) | 0.804 (MIXED ours whole / arm kernel) | - | 1147.9 | 643.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 666.0 | 666.0..666.0 | 1 | 0.671 | - | - | 666.0 | - (stored kernel) | 0.671 (MIXED ours whole / arm kernel) | - | 1100.0 | 643.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 573.6 | 573.6..573.6 | 1 | 0.779 | - | - | 573.6 | - (stored kernel) | 0.779 (MIXED ours whole / arm kernel) | - | 1151.3 | 643.7 | accuracy=0.971842 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 834.6 | 834.6..834.6 | 1 | 0.536 | - | - | 834.6 | - (stored kernel) | 0.536 (MIXED ours whole / arm kernel) | - | 1461.1 | 340.9 | accuracy=0.971951 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 717.1 | 717.1..717.1 | 1 | 0.623 | - | - | 717.1 | - (stored kernel) | 0.623 (MIXED ours whole / arm kernel) | - | 1514.2 | 340.9 | accuracy=0.971951 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 733.2 ms) = 0.611; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 552.2 ms) = 0.811. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.7 | 447.7..447.7 | 1 | - | - | 447.7 | - | - (stored whole) | - | - | - | - | accuracy=0.865723, logloss=0.303537 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 552.2 | 552.2..552.2 | 1 | 0.811 | - | - | 552.2 | - (stored kernel) | 0.811 (MIXED ours whole / arm kernel) | - | 1098.6 | 643.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 552.3 | 552.3..552.3 | 1 | 0.811 | - | - | 552.3 | - (stored kernel) | 0.811 (MIXED ours whole / arm kernel) | - | 1150.5 | 643.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 589.2 | 589.2..589.2 | 1 | 0.760 | - | - | 589.2 | - (stored kernel) | 0.760 (MIXED ours whole / arm kernel) | - | 1099.9 | 643.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 582.4 | 582.4..582.4 | 1 | 0.769 | - | - | 582.4 | - (stored kernel) | 0.769 (MIXED ours whole / arm kernel) | - | 1151.5 | 643.7 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 963.1 | 963.1..963.1 | 1 | 0.465 | - | - | 963.1 | - (stored kernel) | 0.465 (MIXED ours whole / arm kernel) | - | 1463.6 | 340.9 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 733.2 | 733.2..733.2 | 1 | 0.611 | - | - | 733.2 | - (stored kernel) | 0.611 (MIXED ours whole / arm kernel) | - | 1515.8 | 340.9 | accuracy=0.865668 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 819.8 ms) = 0.544; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 538.4 ms) = 0.829. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 446.3 | 446.3..446.3 | 1 | - | - | 446.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.982337, rmse=0.153975 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 538.4 | 538.4..538.4 | 1 | 0.829 | - | - | 538.4 | - (stored kernel) | 0.829 (MIXED ours whole / arm kernel) | - | 1113.9 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 793.5 | 793.5..793.5 | 1 | 0.563 | - | - | 793.5 | - (stored kernel) | 0.563 (MIXED ours whole / arm kernel) | - | 1163.8 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 695.2 | 695.2..695.2 | 1 | 0.642 | - | - | 695.2 | - (stored kernel) | 0.642 (MIXED ours whole / arm kernel) | - | 1115.2 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 708.3 | 708.3..708.3 | 1 | 0.630 | - | - | 708.3 | - (stored kernel) | 0.630 (MIXED ours whole / arm kernel) | - | 1167.1 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 874.5 | 874.5..874.5 | 1 | 0.510 | - | - | 874.5 | - (stored kernel) | 0.510 (MIXED ours whole / arm kernel) | - | 1508.6 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 819.8 | 819.8..819.8 | 1 | 0.544 | - | - | 819.8 | - (stored kernel) | 0.544 (MIXED ours whole / arm kernel) | - | 1560.1 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 754.4 ms) = 0.593; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 626.5 ms) = 0.713. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.0 | 447.0..447.0 | 1 | - | - | 447.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.747875, rmse=0.544554 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 626.5 | 626.5..626.5 | 1 | 0.713 | - | - | 626.5 | - (stored kernel) | 0.713 (MIXED ours whole / arm kernel) | - | 1113.9 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 639.5 | 639.5..639.5 | 1 | 0.699 | - | - | 639.5 | - (stored kernel) | 0.699 (MIXED ours whole / arm kernel) | - | 1164.4 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 591.0 | 591.0..591.0 | 1 | 0.756 | - | - | 591.0 | - (stored kernel) | 0.756 (MIXED ours whole / arm kernel) | - | 1114.9 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 788.3 | 788.3..788.3 | 1 | 0.567 | - | - | 788.3 | - (stored kernel) | 0.567 (MIXED ours whole / arm kernel) | - | 1166.9 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 754.4 | 754.4..754.4 | 1 | 0.593 | - | - | 754.4 | - (stored kernel) | 0.593 (MIXED ours whole / arm kernel) | - | 1509.2 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1058.5 | 1058.5..1058.5 | 1 | 0.422 | - | - | 1058.5 | - (stored kernel) | 0.422 (MIXED ours whole / arm kernel) | - | 1559.7 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### incremental-pca / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 864.1 | 864.1..864.1 | 1 | - | - | 864.1 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=1.000000 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 17717.5 | 17717.5..17717.5 | 1 | 0.049 | - | 17717.5 | 17717.5 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 2303.9 | - | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1618.1 | 1618.1..1618.1 | 1 | 0.534 | - | 2362.7 | 1618.1 | 744.68 (upload_ms_untimed) | 0.366 (whole/whole) | - | 2995.7 | 1314.0 | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### incremental-pca / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 107.6 | 107.6..107.6 | 1 | - | - | 107.6 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=0.999995 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 669.5 | 669.5..669.5 | 1 | 0.161 | - | 669.5 | 669.5 | 0.00 (cpu-arm) | 0.161 (whole/whole) | - | 307.8 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 226.3 | 226.3..226.3 | 1 | 0.476 | - | 291.9 | 226.3 | 65.57 (upload_ms_untimed) | 0.369 (whole/whole) | - | 1176.2 | 518.0 | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0128)`, ran on NVIDIA L40S (nv2, RunPod) job v0128

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 756.9 | 756.9..756.9 | 1 | - | - | 756.9 | - | - (stored whole) | - | - | - | - | recall_at_10=0.802100 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0128 2026-10-11; identity vs amd-mi325x: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host f003873fb257) | cuvs 26.08.01 | opponent | 1111.8 | 1111.8..1111.8 | 1 | 0.681 | - | 1455.7 | 1111.8 | 343.95 (upload_ms_untimed) | 0.520 (whole/whole) | - | 1755.5 | 820.0 | recall_at_10=0.798425 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, faiss-cpu: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1020)`, ran on NVIDIA L40S (nv2, RunPod) job v1020

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 100.1 | 100.1..100.1 | 1 | - | - | 100.1 | - | - (stored whole) | - | - | - | - | recall_at_10=0.981250 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1020 2026-10-09; identity vs amd-mi325x: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 67512.8 | 67512.8..67512.8 | 1 | 0.001 | - | 67512.8 | 67512.8 | 0.00 (cpu-arm) | 0.001 (whole/whole) | - | 135.3 | - | recall_at_10=0.979450 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host f003873fb257) | cuvs 26.08.01 | opponent | 312.8 | 312.8..312.8 | 1 | 0.320 | - | 339.7 | 312.8 | 26.91 (upload_ms_untimed) | 0.295 (whole/whole) | - | 979.5 | 468.0 | recall_at_10=0.977950 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0128)`, ran on NVIDIA L40S (nv2, RunPod) job v0128

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 753.5 | 753.5..753.5 | 1 | - | - | 753.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.994575 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0128 2026-10-11; identity vs amd-mi325x: MATCH) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host f003873fb257) | cuvs 26.08.01 | opponent | 1037.5 | 1037.5..1037.5 | 1 | 0.726 | - | 1383.2 | 1037.5 | 345.69 (upload_ms_untimed) | 0.545 (whole/whole) | - | 1796.4 | 820.0 | recall_at_10=0.995050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, faiss-cpu: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 349.6 | 349.6..349.6 | 1 | - | - | 349.6 | - | - (stored whole) | - | - | - | - | recall_at_10=0.999675 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 70811.7 | 70811.7..70811.7 | 1 | 0.005 | - | 70811.7 | 70811.7 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 153.2 | - | recall_at_10=0.999375 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host f003873fb257) | cuvs 26.08.01 | opponent | 285.6 | 285.6..285.6 | 1 | 1.224 | - | 312.8 | 285.6 | 27.19 (upload_ms_untimed) | 1.118 (whole/whole) | - | 1012.9 | 468.0 | recall_at_10=0.998975 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-sq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0318)`, ran on NVIDIA L40S (nv, RunPod) job n0318

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 155.5 | 155.5..155.5 | 1 | - | - | 155.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.934975 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0318 2026-10-08; identity vs the amd columns: n/a) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 4836.7 | 4836.7..4836.7 | 1 | 0.032 | - | 4836.7 | 4836.7 | 0.00 (cpu-arm) | 0.032 (whole/whole) | - | 124.2 | - | recall_at_10=0.857050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host f003873fb257) | cuvs 26.08.01 | opponent | 138.8 | 138.8..138.8 | 1 | 1.120 | - | 166.0 | 138.8 | 27.20 (upload_ms_untimed) | 0.937 (whole/whole) | - | 946.4 | 466.0 | recall_at_10=0.779425 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 210.2 | 210.2..210.2 | 1 | - | - | 210.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 5641.8 | 5641.8..5641.8 | 1 | 0.037 | - | 5641.8 | 5641.8 | 0.00 (cpu-arm) | 0.037 (whole/whole) | - | 1459.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 640.0 | 640.0..640.0 | 1 | 0.328 | - | 1426.8 | 640.0 | 786.73 (upload_ms_untimed) | 0.147 (whole/whole) | - | 3006.1 | 1442.0 | max_abs_diff_vs_sklearn=1.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.0006747 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 9.0 | 9.0..9.0 | 1 | - | - | 9.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 173.1 | 173.1..173.1 | 1 | 0.052 | - | 173.1 | 173.1 | 0.00 (cpu-arm) | 0.052 (whole/whole) | - | 263.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 24.6 | 24.6..24.6 | 1 | 0.365 | - | 84.8 | 24.6 | 60.27 (upload_ms_untimed) | 0.106 (whole/whole) | - | 928.9 | 488.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0110)`, ran on NVIDIA L40S (nv2, RunPod) job v0110

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3928.3 | 3928.3..3928.3 | 1 | - | - | 3928.3 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0110 2026-10-11; identity vs amd-mi325x: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 25374.0 | 25374.0..25374.0 | 1 | 0.155 | - | 25374.0 | 25374.0 | 0.00 (cpu-arm) | 0.155 (whole/whole) | - | 2183.9 | - | rel_error_vs_exact=8.574e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 40992.8 | 40992.8..40992.8 | 1 | 0.096 | - | 40999.0 | 40992.8 | 6.26 (upload_ms_untimed) | 0.096 (whole/whole) | - | 2014.8 | 438.0 | rel_error_vs_exact=0.030429 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on NVIDIA L40S (nv, RunPod) job n0583

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 319.5 | 319.5..319.5 | 1 | - | - | 319.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.056e-07 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 9341.1 | 9341.1..9341.1 | 1 | 0.034 | - | 9341.1 | 9341.1 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 397.1 | - | rel_error_vs_exact=8.154e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 559.8 | 559.8..559.8 | 1 | 0.571 | - | 564.4 | 559.8 | 4.67 (upload_ms_untimed) | 0.566 (whole/whole) | - | 1010.6 | 438.0 | rel_error_vs_exact=7.647e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0104)`, ran on NVIDIA L40S (nv2, RunPod) job v0104

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 51.7 | 51.7..51.7 | 1 | - | - | 51.7 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0104 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 23.2 | 23.2..23.2 | 1 | 2.230 | - | 23.2 | 23.2 | 0.00 (cpu-arm) | 2.230 (whole/whole) | - | 3499.1 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1020)`, ran on NVIDIA L40S (nv2, RunPod) job v1020

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 2.7 | 2.7..2.7 | 1 | - | - | 2.7 | - | - (stored whole) | - | - | - | - | masked_rmse=6.151696 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1020 2026-10-09; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 6.0 | 6.0..6.0 | 1 | 0.452 | - | 6.0 | 6.0 | 0.00 (cpu-arm) | 0.452 (whole/whole) | - | 1952.7 | - | masked_rmse=5.263919 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 51.5 | 51.5..51.5 | 1 | - | - | 51.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x16 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 92.9 | 92.9..92.9 | 1 | 0.554 | - | 92.9 | 92.9 | 0.00 (cpu-arm) | 0.554 (whole/whole) | - | 672.8 | - | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 18.8 | 18.8..18.8 | 1 | 2.738 | - | 48.6 | 18.8 | 29.79 (upload_ms_untimed) | 1.059 (whole/whole) | - | 1267.1 | 558.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x16, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 862.7 | 862.7..862.7 | 1 | - | - | 862.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x259 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 182.8 | 182.8..182.8 | 1 | 4.719 | - | 182.8 | 182.8 | 0.00 (cpu-arm) | 4.719 (whole/whole) | - | 6590.6 | - | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 116.4 | 116.4..116.4 | 1 | 7.412 | - | 146.9 | 116.4 | 30.50 (upload_ms_untimed) | 5.873 (whole/whole) | - | 3299.7 | 1560.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x259, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0041)`, ran on NVIDIA L40S (nv2, RunPod) job v0041

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@15c76449e) | identical | 5.7 | 5.7..5.7 | 1 | - | - | 5.7 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@15c76449e nv2/v0041 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 28.5 | 28.5..28.5 | 1 | 0.202 | - | 28.5 | 28.5 | 0.00 (cpu-arm) | 0.202 (whole/whole) | - | 308.0 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 16.5 | 16.5..16.5 | 1 | 0.347 | - | 46.7 | 16.5 | 30.17 (upload_ms_untimed) | 0.123 (whole/whole) | - | 1003.1 | 490.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0041)`, ran on NVIDIA L40S (nv2, RunPod) job v0041

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@15c76449e) | identical | 6.0 | 6.0..6.0 | 1 | - | - | 6.0 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@15c76449e nv2/v0041 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 110.3 | 110.3..110.3 | 1 | 0.055 | - | 110.3 | 110.3 | 0.00 (cpu-arm) | 0.055 (whole/whole) | - | 295.4 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 7.8 | 7.8..7.8 | 1 | 0.776 | - | 37.8 | 7.8 | 30.04 (upload_ms_untimed) | 0.159 (whole/whole) | - | 990.3 | 472.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lamb / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on NVIDIA L40S (nv, RunPod) job n0526

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 23.8 | 23.8..23.8 | 1 | - | - | 23.8 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-lamb-synthetic-def/lamb-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 337.9 | 337.9..337.9 | 1 | - | - | 337.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328699, rmse=0.684414 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 922.6 | 922.6..922.6 | 1 | 0.366 | - | 922.6 | 922.6 | 0.00 (cpu-arm) | 0.366 (whole/whole) | - | 1975.3 | - | finite=True, r2=-4.245e+13, rmse=5.442e+06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 64.2 | 64.2..64.2 | 1 | 5.260 | - | 794.0 | 64.2 | 729.77 (upload_ms_untimed) | 0.426 (whole/whole) | - | 2966.9 | 1374.0 | finite=True, r2=0.328088, rmse=0.684726 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.1 | 3.1..3.1 | 1 | - | - | 3.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805051 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 50.8 | 50.8..50.8 | 1 | 0.060 | - | 50.8 | 50.8 | 0.00 (cpu-arm) | 0.060 (whole/whole) | - | 292.2 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 5.5 | 5.5..5.5 | 1 | 0.560 | - | 66.0 | 5.5 | 60.58 (upload_ms_untimed) | 0.046 (whole/whole) | - | 1043.5 | 498.0 | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 428.9 | 428.9..428.9 | 1 | - | - | 428.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.325506, rmse=0.686040 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 97412.0 | 97412.0..97412.0 | 1 | 0.004 | - | 97412.0 | 97412.0 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 8911.4 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0063)`, ran on NVIDIA L40S (nv2, RunPod) job v0063

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@2afa942e3) | identical | 18.0 | 18.0..18.0 | 1 | - | - | 18.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909038, rmse=4.803593 | - | main board, one scored run | - | ok (main@2afa942e3 nv2/v0063 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 313.1 | 313.1..313.1 | 1 | 0.058 | - | 313.1 | 313.1 | 0.00 (cpu-arm) | 0.058 (whole/whole) | - | 610.4 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 90.5 | 90.5..90.5 | 1 | - | - | 90.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310834, rmse=0.693461 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 904.7 | 904.7..904.7 | 1 | 0.100 | - | 904.7 | 904.7 | 0.00 (cpu-arm) | 0.100 (whole/whole) | - | 1969.5 | - | finite=True, r2=0.310816, rmse=0.693470 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0075)`, ran on NVIDIA L40S (nv2, RunPod) job v0075

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@6cd6bed7f) | identical | 3.2 | 3.2..3.2 | 1 | - | - | 3.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908998, rmse=4.804663 | - | main board, one scored run | - | ok (main@6cd6bed7f nv2/v0075 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 62.7 | 62.7..62.7 | 1 | 0.051 | - | 62.7 | 62.7 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 296.3 | - | finite=True, r2=0.908998, rmse=4.804667 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 0.732; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.6 ms) = 0.715. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1080/work-layernorm-synthetic-def/layernorm-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.715 | - | - | 1.6 | - (stored kernel) | 0.715 (MIXED ours whole / arm kernel) | - | 721.3 | 320.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.635 | - | - | 1.7 | - (stored kernel) | 0.635 (MIXED ours whole / arm kernel) | - | 967.8 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.731 | - | - | 1.5 | - (stored kernel) | 0.731 (MIXED ours whole / arm kernel) | - | 713.6 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.764 | - | - | 1.5 | - (stored kernel) | 0.764 (MIXED ours whole / arm kernel) | - | 916.0 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.732 | - | - | 1.5 | - (stored kernel) | 0.732 (MIXED ours whole / arm kernel) | - | 721.0 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.707 | - | - | 1.6 | - (stored kernel) | 0.707 (MIXED ours whole / arm kernel) | - | 902.0 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'elementwise_affine': True, 'eps': 1e-05, 'normalized_shape': 1024}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0607)`, ran on NVIDIA L40S (nv, RunPod) job n0607

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 309.9 | 309.9..309.9 | 1 | - | - | 309.9 | - | - (stored whole) | - | - | - | - | accuracy=0.912830, logloss=0.235875 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0607 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 11501.6 | 11501.6..11501.6 | 1 | 0.027 | - | 11501.6 | 11501.6 | 0.00 (cpu-arm) | 0.027 (whole/whole) | - | 4591.9 | - | accuracy=0.899510, logloss=0.438978 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0607)`, ran on NVIDIA L40S (nv, RunPod) job n0607

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 27.9 | 27.9..27.9 | 1 | - | - | 27.9 | - | - (stored whole) | - | - | - | - | accuracy=0.762530, logloss=0.539763 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0607 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 398.4 | 398.4..398.4 | 1 | 0.070 | - | 398.4 | 398.4 | 0.00 (cpu-arm) | 0.070 (whole/whole) | - | 462.6 | - | accuracy=0.762530, logloss=0.539767 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lion / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on NVIDIA L40S (nv, RunPod) job n0526

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32: - (no completed torch fp32 arm). Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 11.0 | 11.0..11.0 | 1 | - | - | 11.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-lion-synthetic-def/lion-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'betas': [0.9, 0.99], 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 178.8 | 178.8..178.8 | 1 | - | - | 178.8 | - | - (stored whole) | - | - | - | - | modularity=0.911187, n_communities=40 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 2943.6 | 2943.6..2943.6 | 1 | 0.061 | - | 2943.6 | 2943.6 | 0.00 (cpu-arm) | 0.061 (whole/whole) | - | 239.0 | - | modularity=0.908460, n_communities=40 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host f003873fb257) | cugraph 26.08.00 | opponent | 260.3 | 260.3..260.3 | 1 | 0.687 | - | 260.3 | 260.3 | 0.01 (upload_ms_untimed) | 0.687 (whole/whole) | - | 1002.0 | 440.0 | modularity=0.909430, n_communities=41 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 38.2 | 38.2..38.2 | 1 | - | - | 38.2 | - | - (stored whole) | - | - | - | - | modularity=0.941953, n_communities=58 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 1444.9 | 1444.9..1444.9 | 1 | 0.026 | - | 1444.9 | 1444.9 | 0.00 (cpu-arm) | 0.026 (whole/whole) | - | 201.6 | - | modularity=0.940781, n_communities=56 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host f003873fb257) | cugraph 26.08.00 | opponent | 146.2 | 146.2..146.2 | 1 | 0.261 | - | 146.2 | 146.2 | 0.01 (upload_ms_untimed) | 0.261 (whole/whole) | - | 902.7 | 436.0 | modularity=0.941795, n_communities=62 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 735.7 ms) = 0.743; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 539.4 ms) = 1.013. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 546.6 | 546.6..546.6 | 1 | - | - | 546.6 | - | - (stored whole) | - | - | - | - | accuracy=0.967068, logloss=0.079720 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 539.4 | 539.4..539.4 | 1 | 1.013 | - | - | 539.4 | - (stored kernel) | 1.013 (MIXED ours whole / arm kernel) | - | 1098.6 | 695.4 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 659.5 | 659.5..659.5 | 1 | 0.829 | - | - | 659.5 | - (stored kernel) | 0.829 (MIXED ours whole / arm kernel) | - | 1149.4 | 695.4 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 568.3 | 568.3..568.3 | 1 | 0.962 | - | - | 568.3 | - (stored kernel) | 0.962 (MIXED ours whole / arm kernel) | - | 1100.2 | 695.4 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 843.6 | 843.6..843.6 | 1 | 0.648 | - | - | 843.6 | - (stored kernel) | 0.648 (MIXED ours whole / arm kernel) | - | 1151.7 | 695.4 | accuracy=0.968696 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 735.7 | 735.7..735.7 | 1 | 0.743 | - | - | 735.7 | - (stored kernel) | 0.743 (MIXED ours whole / arm kernel) | - | 1466.9 | 367.2 | accuracy=0.968913 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 824.4 | 824.4..824.4 | 1 | 0.663 | - | - | 824.4 | - (stored kernel) | 0.663 (MIXED ours whole / arm kernel) | - | 1518.3 | 367.2 | accuracy=0.968913 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 707.9 ms) = 0.772; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 531.6 ms) = 1.028. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 546.5 | 546.5..546.5 | 1 | - | - | 546.5 | - | - (stored whole) | - | - | - | - | accuracy=0.870443, logloss=0.297146 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 531.6 | 531.6..531.6 | 1 | 1.028 | - | - | 531.6 | - (stored kernel) | 1.028 (MIXED ours whole / arm kernel) | - | 1097.4 | 695.4 | accuracy=0.868218 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 643.1 | 643.1..643.1 | 1 | 0.850 | - | - | 643.1 | - (stored kernel) | 0.850 (MIXED ours whole / arm kernel) | - | 1150.2 | 695.4 | accuracy=0.868218 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 712.0 | 712.0..712.0 | 1 | 0.768 | - | - | 712.0 | - (stored kernel) | 0.768 (MIXED ours whole / arm kernel) | - | 1100.3 | 695.4 | accuracy=0.868164 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 560.1 | 560.1..560.1 | 1 | 0.976 | - | - | 560.1 | - (stored kernel) | 0.976 (MIXED ours whole / arm kernel) | - | 1150.7 | 695.4 | accuracy=0.868164 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 707.9 | 707.9..707.9 | 1 | 0.772 | - | - | 707.9 | - (stored kernel) | 0.772 (MIXED ours whole / arm kernel) | - | 1466.7 | 367.2 | accuracy=0.868327 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 853.0 | 853.0..853.0 | 1 | 0.641 | - | - | 853.0 | - (stored kernel) | 0.641 (MIXED ours whole / arm kernel) | - | 1517.8 | 367.2 | accuracy=0.868327 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 729.3 ms) = 0.748; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 574.8 ms) = 0.949. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 545.6 | 545.6..545.6 | 1 | - | - | 545.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.979398, rmse=0.166295 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 729.3 | 729.3..729.3 | 1 | 0.748 | - | - | 729.3 | - (stored kernel) | 0.748 (MIXED ours whole / arm kernel) | - | 1109.0 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 574.8 | 574.8..574.8 | 1 | 0.949 | - | - | 574.8 | - (stored kernel) | 0.949 (MIXED ours whole / arm kernel) | - | 1159.4 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 656.8 | 656.8..656.8 | 1 | 0.831 | - | - | 656.8 | - (stored kernel) | 0.831 (MIXED ours whole / arm kernel) | - | 1109.9 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 627.3 | 627.3..627.3 | 1 | 0.870 | - | - | 627.3 | - (stored kernel) | 0.870 (MIXED ours whole / arm kernel) | - | 1162.0 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 729.3 | 729.3..729.3 | 1 | 0.748 | - | - | 729.3 | - (stored kernel) | 0.748 (MIXED ours whole / arm kernel) | - | 1511.9 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 829.2 | 829.2..829.2 | 1 | 0.658 | - | - | 829.2 | - (stored kernel) | 0.658 (MIXED ours whole / arm kernel) | - | 1563.9 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 742.3 ms) = 0.735; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 606.4 ms) = 0.900. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 545.8 | 545.8..545.8 | 1 | - | - | 545.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.754306, rmse=0.537563 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 646.5 | 646.5..646.5 | 1 | 0.844 | - | - | 646.5 | - (stored kernel) | 0.844 (MIXED ours whole / arm kernel) | - | 1109.0 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 606.4 | 606.4..606.4 | 1 | 0.900 | - | - | 606.4 | - (stored kernel) | 0.900 (MIXED ours whole / arm kernel) | - | 1158.0 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 585.0 | 585.0..585.0 | 1 | 0.933 | - | - | 585.0 | - (stored kernel) | 0.933 (MIXED ours whole / arm kernel) | - | 1110.7 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 633.7 | 633.7..633.7 | 1 | 0.861 | - | - | 633.7 | - (stored kernel) | 0.861 (MIXED ours whole / arm kernel) | - | 1160.8 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 742.3 | 742.3..742.3 | 1 | 0.735 | - | - | 742.3 | - (stored kernel) | 0.735 (MIXED ours whole / arm kernel) | - | 1512.9 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 785.0 | 785.0..785.0 | 1 | 0.695 | - | - | 785.0 | - (stored kernel) | 0.695 (MIXED ours whole / arm kernel) | - | 1562.6 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 311.7 | 311.7..311.7 | 1 | - | - | 311.7 | - | - (stored whole) | - | - | - | - | relative_residual=0.849956 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 7725.5 | 7725.5..7725.5 | 1 | 0.040 | - | 7725.5 | 7725.5 | 0.00 (cpu-arm) | 0.040 (whole/whole) | - | 4348.1 | - | relative_residual=0.876581 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 220.9 | 220.9..220.9 | 1 | 1.411 | - | - | 220.9 | - (stored kernel) | 1.411 (MIXED ours whole / arm kernel) | - | 1758.5 | 3516.2 | relative_residual=nan | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 244.5 | 244.5..244.5 | 1 | 1.275 | - | - | 244.5 | - (stored kernel) | 1.275 (MIXED ours whole / arm kernel) | - | 2721.0 | 9158.0 | relative_residual=0.849957 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 7.5 | 7.5..7.5 | 1 | - | - | 7.5 | - | - (stored whole) | - | - | - | - | relative_residual=0.756366 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 183.1 | 183.1..183.1 | 1 | 0.041 | - | 183.1 | 183.1 | 0.00 (cpu-arm) | 0.041 (whole/whole) | - | 279.6 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 7.3 | 7.3..7.3 | 1 | 1.032 | - | - | 7.3 | - (stored kernel) | 1.032 (MIXED ours whole / arm kernel) | - | 888.4 | 1122.8 | relative_residual=0.756366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 9.1 | 9.1..9.1 | 1 | 0.824 | - | - | 9.1 | - (stored kernel) | 0.824 (MIXED ours whole / arm kernel) | - | 713.5 | 2778.0 | relative_residual=0.756366 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0072)`, ran on NVIDIA L40S (nv2, RunPod) job v0072

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 359.4 | 359.4..359.4 | 1 | - | - | 359.4 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@c69768593 nv2/v0072 2026-10-11; identity vs amd-mi325x: MATCH) |
| scipy-cpu | scipy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scipy 1.17.1 | opponent | 5417.0 | 5417.0..5417.0 | 1 | 0.066 | - | 5417.0 | 5417.0 | 0.00 (cpu-arm) | 0.066 (whole/whole) | - | 600.1 | - | relative_residual=4.275e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 36.2 | 36.2..36.2 | 1 | 9.940 | - | - | 36.2 | - (stored kernel) | 9.940 (MIXED ours whole / arm kernel) | - | 802.5 | 528.3 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 40.1 | 40.1..40.1 | 1 | 8.971 | - | - | 40.1 | - (stored kernel) | 8.971 (MIXED ours whole / arm kernel) | - | 864.5 | 1108.0 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, scipy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0072)`, ran on NVIDIA L40S (nv2, RunPod) job v0072

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 352.0 | 352.0..352.0 | 1 | - | - | 352.0 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@c69768593 nv2/v0072 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 10448.5 | 10448.5..10448.5 | 1 | 0.034 | - | 10448.5 | 10448.5 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 1360.8 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 36.2 | 36.2..36.2 | 1 | 9.725 | - | - | 36.2 | - (stored kernel) | 9.725 (MIXED ours whole / arm kernel) | - | 802.4 | 528.3 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 36.0 | 36.0..36.0 | 1 | 9.782 | - | - | 36.0 | - (stored kernel) | 9.782 (MIXED ours whole / arm kernel) | - | 792.9 | 984.0 | relative_residual=3.386e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 38.7 | 38.7..38.7 | 1 | - | - | 38.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 153.4 | 153.4..153.4 | 1 | 0.252 | - | 153.4 | 153.4 | 0.00 (cpu-arm) | 0.252 (whole/whole) | - | 2214.7 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 29.6 | 29.6..29.6 | 1 | 1.306 | - | 793.2 | 29.6 | 763.60 (upload_ms_untimed) | 0.049 (whole/whole) | - | 3149.0 | 1442.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 2.5 | 2.5..2.5 | 1 | - | - | 2.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 41.8 | 41.8..41.8 | 1 | 0.061 | - | 41.8 | 41.8 | 0.00 (cpu-arm) | 0.061 (whole/whole) | - | 301.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 3.9 | 3.9..3.9 | 1 | 0.645 | - | 66.8 | 3.9 | 62.87 (upload_ms_untimed) | 0.038 (whole/whole) | - | 870.6 | 488.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0082)`, ran on NVIDIA L40S (nv2, RunPod) job v0082

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.7 ms) = 0.741; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.2 ms) = 1.020. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 1.2 | 1.2..1.2 | 1 | - | - | 1.2 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0082 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 1.020 | - | - | 1.2 | - (stored kernel) | 1.020 (MIXED ours whole / arm kernel) | - | 691.9 | 288.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 0.955 | - | - | 1.3 | - (stored kernel) | 0.955 (MIXED ours whole / arm kernel) | - | 933.1 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.822 | - | - | 1.5 | - (stored kernel) | 0.822 (MIXED ours whole / arm kernel) | - | 691.4 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.796 | - | - | 1.6 | - (stored kernel) | 0.796 (MIXED ours whole / arm kernel) | - | 883.0 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 0.490 | - | - | 2.5 | - (stored kernel) | 0.490 (MIXED ours whole / arm kernel) | - | 691.6 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.741 | - | - | 1.7 | - (stored kernel) | 0.741 (MIXED ours whole / arm kernel) | - | 877.8 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.9 ms) = 0.775; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.9 ms) = 0.757. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.5 | 1.5..1.5 | 1 | - | - | 1.5 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.757 | - | - | 1.9 | - (stored kernel) | 0.757 (MIXED ours whole / arm kernel) | - | 707.4 | 639.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.725 | - | - | 2.0 | - (stored kernel) | 0.725 (MIXED ours whole / arm kernel) | - | 1208.8 | 552.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.859 | - | - | 1.7 | - (stored kernel) | 0.859 (MIXED ours whole / arm kernel) | - | 706.1 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 0.577 | - | - | 2.5 | - (stored kernel) | 0.577 (MIXED ours whole / arm kernel) | - | 900.9 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.775 | - | - | 1.9 | - (stored kernel) | 0.775 (MIXED ours whole / arm kernel) | - | 707.4 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 0.341 | - | - | 4.3 | - (stored kernel) | 0.341 (MIXED ours whole / arm kernel) | - | 894.0 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0106)`, ran on NVIDIA L40S (nv2, RunPod) job v0106

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 518.6 | 518.6..518.6 | 1 | - | - | 518.6 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.116696 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0106 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 242.5 | 242.5..242.5 | 1 | 2.138 | - | 242.5 | 242.5 | 0.00 (cpu-arm) | 2.138 (whole/whole) | - | 1178.0 | - | ari_vs_ours=0.622446, n_clusters=8, silhouette=0.111849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### minibatch-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0106)`, ran on NVIDIA L40S (nv2, RunPod) job v0106

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 48.7 | 48.7..48.7 | 1 | - | - | 48.7 | - | - (stored whole) | - | - | - | - | n_clusters=8, silhouette=0.138060 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0106 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 50.8 | 50.8..50.8 | 1 | 0.958 | - | 50.8 | 50.8 | 0.00 (cpu-arm) | 0.958 (whole/whole) | - | 276.0 | - | ari_vs_ours=0.525151, n_clusters=8, silhouette=0.165473 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### moe / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0100)`, ran on NVIDIA L40S (nv2, RunPod) job v0100

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 4.5 ms) = 3.157; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 11.9 ms) = 1.190. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 14.1 | 14.1..14.1 | 1 | - | - | 14.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v0100/work-moe-synthetic-def/moe-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0100 2026-10-11; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 12.6 | 12.6..12.6 | 1 | 1.122 | - | - | 12.6 | - (stored kernel) | 1.122 (MIXED ours whole / arm kernel) | - | 998.2 | 469.1 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 11.9 | 11.9..11.9 | 1 | 1.190 | - | - | 11.9 | - (stored kernel) | 1.190 (MIXED ours whole / arm kernel) | - | 1258.9 | 446.7 | max_rel_diff_vs_torch_eager_fp32=0.026193, rel_fro_vs_torch_eager_fp32=1.548e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 6.3 | 6.3..6.3 | 1 | 2.245 | - | - | 6.3 | - (stored kernel) | 2.245 (MIXED ours whole / arm kernel) | - | 990.2 | 469.1 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 5.2 | 5.2..5.2 | 1 | 2.704 | - | - | 5.2 | - (stored kernel) | 2.704 (MIXED ours whole / arm kernel) | - | 1206.2 | 478.7 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.5 | 4.5..4.5 | 1 | 3.157 | - | - | 4.5 | - (stored kernel) | 3.157 (MIXED ours whole / arm kernel) | - | 1115.4 | 574.3 | max_rel_diff_vs_torch_eager_fp32=22834.612745, rel_fro_vs_torch_eager_fp32=0.055222 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.6 | 4.6..4.6 | 1 | 3.081 | - | - | 4.6 | - (stored kernel) | 3.081 (MIXED ours whole / arm kernel) | - | 1386.3 | 433.6 | max_rel_diff_vs_torch_eager_fp32=22851.987745, rel_fro_vs_torch_eager_fp32=0.055199 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 112.8 | 112.8..112.8 | 1 | - | - | 112.8 | - | - (stored whole) | - | - | - | - | accuracy=0.853620, logloss=3.628565 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 271.1 | 271.1..271.1 | 1 | 0.416 | - | 271.1 | 271.1 | 0.00 (cpu-arm) | 0.416 (whole/whole) | - | 3913.9 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 56.8 | 56.8..56.8 | 1 | 1.984 | - | 793.5 | 56.8 | 736.63 (upload_ms_untimed) | 0.142 (whole/whole) | - | 3888.1 | 1372.0 | accuracy=0.853620, logloss=3.628599 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 30.2 | 30.2..30.2 | 1 | - | - | 30.2 | - | - (stored whole) | - | - | - | - | accuracy=0.723160, logloss=0.590725 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 69.9 | 69.9..69.9 | 1 | 0.432 | - | 69.9 | 69.9 | 0.00 (cpu-arm) | 0.432 (whole/whole) | - | 436.4 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 12.4 | 12.4..12.4 | 1 | 2.433 | - | 74.1 | 12.4 | 61.64 (upload_ms_untimed) | 0.408 (whole/whole) | - | 1054.1 | 496.0 | accuracy=0.723160, logloss=0.590750 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on NVIDIA L40S (nv, RunPod) job n0532

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 56.2 | 56.2..56.2 | 1 | - | - | 56.2 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559529 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 454.4 | 454.4..454.4 | 1 | 0.124 | - | 454.4 | 454.4 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 108.0 | 108.0..108.0 | 1 | 0.520 | - | 1546.2 | 108.0 | 1438.14 (upload_ms_untimed) | 0.036 (whole/whole) | - | 4688.2 | 1946.0 | accuracy=0.983067, logloss=0.559524 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nadam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 21.8 ms) = 0.496. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 10.8 | 10.8..10.8 | 1 | - | - | 10.8 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 21.8 | 21.8..21.8 | 1 | 0.496 | - | - | 21.8 | - (stored kernel) | 0.496 (MIXED ours whole / arm kernel) | - | 959.0 | 960.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 171.1 | 171.1..171.1 | 1 | 0.063 | - | - | 171.1 | - (stored kernel) | 0.063 (MIXED ours whole / arm kernel) | - | 1429.2 | 896.0 | rel_fro_vs_torch_eager_fp32=1.498e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nearest-centroid / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0102)`, ran on NVIDIA L40S (nv2, RunPod) job v0102

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 46.0 | 46.0..46.0 | 1 | - | - | 46.0 | - | - (stored whole) | - | - | - | - | accuracy=0.852610, logloss=4.299485 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0102 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1024.1 | 1024.1..1024.1 | 1 | 0.045 | - | 1024.1 | 1024.1 | 0.00 (cpu-arm) | 0.045 (whole/whole) | - | 4523.4 | - | accuracy=0.852610, logloss=4.117692 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nearest-centroid / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0102)`, ran on NVIDIA L40S (nv2, RunPod) job v0102

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 9.2 | 9.2..9.2 | 1 | - | - | 9.2 | - | - (stored whole) | - | - | - | - | accuracy=0.666700, logloss=0.781905 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0102 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 260.8 | 260.8..260.8 | 1 | 0.035 | - | 260.8 | 260.8 | 0.00 (cpu-arm) | 0.035 (whole/whole) | - | 457.8 | - | accuracy=0.666750, logloss=0.781690 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 39.2 | 39.2..39.2 | 1 | 0.004 | - | 39.2 | 39.2 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1459.3 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 2.5 | 2.5..2.5 | 1 | 0.069 | - | 806.6 | 2.5 | 804.10 (upload_ms_untimed) | 0.0002094 (whole/whole) | - | 3027.7 | 1450.0 | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x220, rel_diff_vs_sklearn=4.428e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2.5 | 2.5..2.5 | 1 | 0.043 | - | 2.5 | 2.5 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 265.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 0.5 | 0.5..0.5 | 1 | 0.240 | - | 62.0 | 0.5 | 61.58 (upload_ms_untimed) | 0.002 (whole/whole) | - | 942.0 | 496.0 | max_abs_diff_vs_sklearn=1.788e-07, output_shape=100000x11, rel_diff_vs_sklearn=2.246e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 30.8 | 30.8..30.8 | 1 | - | - | 30.8 | - | - (stored whole) | - | - | - | - | output_shape=100000x119 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 93.8 | 93.8..93.8 | 1 | 0.328 | - | 93.8 | 93.8 | 0.00 (cpu-arm) | 0.328 (whole/whole) | - | 531.2 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 50.9 | 50.9..50.9 | 1 | 0.605 | - | 81.4 | 50.9 | 30.48 (upload_ms_untimed) | 0.378 (whole/whole) | - | 1227.9 | 524.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x119, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 23.7 | 23.7..23.7 | 1 | - | - | 23.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x508 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 48.4 | 48.4..48.4 | 1 | 0.489 | - | 48.4 | 48.4 | 0.00 (cpu-arm) | 0.489 (whole/whole) | - | 1399.8 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 40.5 | 40.5..40.5 | 1 | 0.585 | - | 71.5 | 40.5 | 30.96 (upload_ms_untimed) | 0.331 (whole/whole) | - | 1696.0 | 656.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x508, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 175.1 | 175.1..175.1 | 1 | - | - | 175.1 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.438855 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 750.7 | 750.7..750.7 | 1 | 0.233 | - | 750.7 | 750.7 | 0.00 (cpu-arm) | 0.233 (whole/whole) | - | 467.7 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 309.6 | 309.6..309.6 | 1 | - | - | 309.6 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.150860 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1772.3 | 1772.3..1772.3 | 1 | 0.175 | - | 1772.3 | 1772.3 | 0.00 (cpu-arm) | 0.175 (whole/whole) | - | 467.1 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 32.8 | 32.8..32.8 | 1 | - | - | 32.8 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 100.3 | 100.3..100.3 | 1 | 0.328 | - | 100.3 | 100.3 | 0.00 (cpu-arm) | 0.328 (whole/whole) | - | 259.8 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0124)`, ran on NVIDIA L40S (nv2, RunPod) job v0124

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 21.0 | 21.0..21.0 | 1 | - | - | 21.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0124 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 48.5 | 48.5..48.5 | 1 | 0.432 | - | 48.5 | 48.5 | 0.00 (cpu-arm) | 0.432 (whole/whole) | - | 240.9 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.4 | 1.4..1.4 | 1 | - | - | 1.4 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 274.0 | 274.0..274.0 | 1 | 0.005 | - | 274.0 | 274.0 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 211.2 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host f003873fb257) | cugraph 26.08.00 | opponent | 4.7 | 4.7..4.7 | 1 | 0.298 | - | 4.7 | 4.7 | 0.01 (upload_ms_untimed) | 0.297 (whole/whole) | - | 1112.3 | 440.0 | l1_vs_networkx=0.010895, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.3 | 1.3..1.3 | 1 | - | - | 1.3 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 189.4 | 189.4..189.4 | 1 | 0.007 | - | 189.4 | 189.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 175.9 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host f003873fb257) | cugraph 26.08.00 | opponent | 4.7 | 4.7..4.7 | 1 | 0.270 | - | 4.8 | 4.7 | 0.05 (upload_ms_untimed) | 0.267 (whole/whole) | - | 974.6 | 438.0 | l1_vs_networkx=0.007172, sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0108)`, ran on NVIDIA L40S (nv2, RunPod) job v0108

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 3872.5 | 3872.5..3872.5 | 1 | - | - | 3872.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0108 2026-10-11; identity vs amd-mi325x: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 90631.6 | 90631.6..90631.6 | 1 | 0.043 | - | 90631.6 | 90631.6 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 1725.8 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1771.4 | 1771.4..1771.4 | 1 | 2.186 | - | 1776.0 | 1771.4 | 4.66 (upload_ms_untimed) | 2.180 (whole/whole) | - | 1864.7 | 438.0 | rel_error_vs_exact=1.497e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0108)`, ran on NVIDIA L40S (nv2, RunPod) job v0108

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 66.3 | 66.3..66.3 | 1 | - | - | 66.3 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0108 2026-10-11; identity vs amd-mi325x: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 133.5 | 133.5..133.5 | 1 | 0.497 | - | 133.5 | 133.5 | 0.00 (cpu-arm) | 0.497 (whole/whole) | - | 476.7 | - | rel_error_vs_exact=1.144e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 324.3 | 324.3..324.3 | 1 | 0.205 | - | 328.7 | 324.3 | 4.46 (upload_ms_untimed) | 0.202 (whole/whole) | - | 899.9 | 438.0 | rel_error_vs_exact=1.655e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x152 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.2 | 3.2..3.2 | 1 | 0.156 | - | 3.2 | 3.2 | 0.00 (cpu-arm) | 0.156 (whole/whole) | - | 1430.3 | - | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 3.2 | 3.2..3.2 | 1 | 0.154 | - | 68.1 | 3.2 | 64.94 (upload_ms_untimed) | 0.007 (whole/whole) | - | 1890.1 | 560.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x152, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.3 | 0.3..0.3 | 1 | - | - | 0.3 | - | - (stored whole) | - | - | - | - | output_shape=100000x77 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2.2 | 2.2..2.2 | 1 | 0.125 | - | 2.2 | 2.2 | 0.00 (cpu-arm) | 0.125 (whole/whole) | - | 369.8 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1.0 | 1.0..1.0 | 1 | 0.280 | - | 63.2 | 1.0 | 62.19 (upload_ms_untimed) | 0.004 (whole/whole) | - | 921.9 | 510.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x77, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0318)`, ran on NVIDIA L40S (nv, RunPod) job n0318

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 120.9 | 120.9..120.9 | 1 | - | - | 120.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0318 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 4294.8 | 4294.8..4294.8 | 1 | 0.028 | - | 4294.8 | 4294.8 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 397.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 2670.2 | 2670.2..2670.2 | 1 | 0.045 | - | 2731.1 | 2670.2 | 60.89 (upload_ms_untimed) | 0.044 (whole/whole) | - | 918.6 | 488.0 | max_abs_diff_vs_sklearn=0.063349, output_shape=100000x11, rel_diff_vs_sklearn=0.0003244 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 880.3 | 880.3..880.3 | 1 | - | - | 880.3 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.485e-07 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 17211.6 | 17211.6..17211.6 | 1 | 0.051 | - | 17211.6 | 17211.6 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 8528.1 | - | relative_gram_difference=2.459e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 227.7 | 227.7..227.7 | 1 | 3.866 | - | - | 227.7 | - (stored kernel) | 3.866 (MIXED ours whole / arm kernel) | - | 1680.6 | 3008.9 | relative_gram_difference=3.787e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 231.1 | 231.1..231.1 | 1 | 3.810 | - | - | 231.1 | - (stored kernel) | 3.810 (MIXED ours whole / arm kernel) | - | 2442.6 | 5160.0 | relative_gram_difference=3.787e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 31.5 | 31.5..31.5 | 1 | - | - | 31.5 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.714e-07 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 444.9 | 444.9..444.9 | 1 | 0.071 | - | 444.9 | 444.9 | 0.00 (cpu-arm) | 0.071 (whole/whole) | - | 475.9 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 7.0 | 7.0..7.0 | 1 | 4.484 | - | - | 7.0 | - (stored kernel) | 4.484 (MIXED ours whole / arm kernel) | - | 806.8 | 168.1 | relative_gram_difference=5.971e-06 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 7.0 | 7.0..7.0 | 1 | 4.465 | - | - | 7.0 | - (stored kernel) | 4.465 (MIXED ours whole / arm kernel) | - | 642.3 | 726.0 | relative_gram_difference=5.971e-06 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile-transformer / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0029)`, ran on NVIDIA L40S (nv2, RunPod) job v0029

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 211.8 | 211.8..211.8 | 1 | - | - | 211.8 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0029 2026-10-11; identity vs the amd columns: n/a) |

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'subsample': 1000000000, 'random_state': 7}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile-transformer / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0029)`, ran on NVIDIA L40S (nv2, RunPod) job v0029

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 9.0 | 9.0..9.0 | 1 | - | - | 9.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0029 2026-10-11; identity vs the amd columns: n/a) |

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'subsample': 1000000000, 'random_state': 7}. Rows: None. Timed: None.

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0328)`, ran on NVIDIA L40S (nv, RunPod) job n0328

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 358.7 | 358.7..358.7 | 1 | - | - | 358.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.039998, rmse=0.851877 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0328 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1.404e+06 | 1.404e+06..1.404e+06 | 1 | 0.0002554 | - | 1.404e+06 | 1.404e+06 | 0.00 (cpu-arm) | 0.0002554 (whole/whole) | - | 7995.7 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0328)`, ran on NVIDIA L40S (nv, RunPod) job n0328

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 234.9 | 234.9..234.9 | 1 | - | - | 234.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.899596, rmse=5.046749 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0328 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 386298.7 | 386298.7..386298.7 | 1 | 0.0006081 | - | 386298.7 | 386298.7 | 0.00 (cpu-arm) | 0.0006081 (whole/whole) | - | 926.3 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 163.7 | 163.7..163.7 | 1 | - | - | 163.7 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.0002359 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1589.1 | 1589.1..1589.1 | 1 | 0.103 | - | 1589.1 | 1589.1 | 0.00 (cpu-arm) | 0.103 (whole/whole) | - | 2079.3 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 66.1 | 66.1..66.1 | 1 | 2.477 | - | - | 66.1 | - (stored kernel) | 2.477 (MIXED ours whole / arm kernel) | - | 1614.2 | 1012.2 | relative_reconstruction_error=0.0002359 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 57.7 | 57.7..57.7 | 1 | - | - | 57.7 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.027197 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1017.5 | 1017.5..1017.5 | 1 | 0.057 | - | 1017.5 | 1017.5 | 0.00 (cpu-arm) | 0.057 (whole/whole) | - | 477.8 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 32.3 | 32.3..32.3 | 1 | 1.784 | - | - | 32.3 | - (stored kernel) | 1.784 (MIXED ours whole / arm kernel) | - | 813.9 | 198.1 | relative_reconstruction_error=0.027197 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0140)`, ran on NVIDIA L40S (nv2, RunPod) job v0140

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 2.7 ms) = 5.756; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 4.8 ms) = 3.203. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@7924718fc) | identical | 15.3 | 15.3..15.3 | 1 | - | - | 15.3 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (mojolearn: MOJOLEARN_VENDOR=cpu; no GPU set is loaded, on request (flat layout)); see /root/lq/out/v0140/work-resnet-block-synthetic-def/resnet-block-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@7924718fc nv2/v0140 2026-10-11; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 6.0 | 6.0..6.0 | 1 | 2.577 | - | - | 6.0 | - (stored kernel) | 2.577 (MIXED ours whole / arm kernel) | - | 926.0 | 450.7 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.8 | 4.8..4.8 | 1 | 3.203 | - | - | 4.8 | - (stored kernel) | 3.203 (MIXED ours whole / arm kernel) | - | 1127.1 | 410.2 | max_rel_diff_vs_torch_eager_fp32=0.818182, rel_fro_vs_torch_eager_fp32=5.381e-07 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.4 | 4.4..4.4 | 1 | 3.488 | - | - | 4.4 | - (stored kernel) | 3.488 (MIXED ours whole / arm kernel) | - | 942.9 | 554.3 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 3.591 | - | - | 4.3 | - (stored kernel) | 3.591 (MIXED ours whole / arm kernel) | - | 1092.2 | 516.8 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.8 | 2.8..2.8 | 1 | 5.482 | - | - | 2.8 | - (stored kernel) | 5.482 (MIXED ours whole / arm kernel) | - | 985.7 | 424.9 | max_rel_diff_vs_torch_eager_fp32=17838.627100, rel_fro_vs_torch_eager_fp32=0.003621 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 2.7 | 2.7..2.7 | 1 | 5.756 | - | - | 2.7 | - (stored kernel) | 5.756 (MIXED ours whole / arm kernel) | - | 1157.4 | 384.4 | max_rel_diff_vs_torch_eager_fp32=18497.318029, rel_fro_vs_torch_eager_fp32=0.003425 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0079)`, ran on NVIDIA L40S (nv2, RunPod) job v0079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 23140.7 | 23140.7..23140.7 | 1 | - | - | 23140.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328684, rmse=0.684422 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0079 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 188507.2 | 188507.2..188507.2 | 1 | 0.123 | - | 188507.2 | 188507.2 | 0.00 (cpu-arm) | 0.123 (whole/whole) | - | 8535.6 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0029)`, ran on NVIDIA L40S (nv2, RunPod) job v0029

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 154.7 | 154.7..154.7 | 1 | - | - | 154.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0029 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2252.8 | 2252.8..2252.8 | 1 | 0.069 | - | 2252.8 | 2252.8 | 0.00 (cpu-arm) | 0.069 (whole/whole) | - | 378.4 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 14.8 ms) = 0.579. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 8.6 | 8.6..8.6 | 1 | - | - | 8.6 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 14.8 | 14.8..14.8 | 1 | 0.579 | - | - | 14.8 | - (stored kernel) | 0.579 (MIXED ours whole / arm kernel) | - | 945.8 | 896.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 121.9 | 121.9..121.9 | 1 | 0.070 | - | - | 121.9 | - (stored kernel) | 0.070 (MIXED ours whole / arm kernel) | - | 1295.6 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 829.7 ms) = 0.294; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 497.4 ms) = 0.491. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 244.0 | 244.0..244.0 | 1 | - | - | 244.0 | - | - (stored whole) | - | - | - | - | accuracy=0.967828, logloss=0.079029 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 497.4 | 497.4..497.4 | 1 | 0.491 | - | - | 497.4 | - (stored kernel) | 0.491 (MIXED ours whole / arm kernel) | - | 1098.5 | 404.2 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 550.6 | 550.6..550.6 | 1 | 0.443 | - | - | 550.6 | - (stored kernel) | 0.443 (MIXED ours whole / arm kernel) | - | 1150.4 | 404.2 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 487.3 | 487.3..487.3 | 1 | 0.501 | - | - | 487.3 | - (stored kernel) | 0.501 (MIXED ours whole / arm kernel) | - | 1099.7 | 404.2 | accuracy=0.953505 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 521.6 | 521.6..521.6 | 1 | 0.468 | - | - | 521.6 | - (stored kernel) | 0.468 (MIXED ours whole / arm kernel) | - | 1151.4 | 404.2 | accuracy=0.953505 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 829.7 | 829.7..829.7 | 1 | 0.294 | - | - | 829.7 | - (stored kernel) | 0.294 (MIXED ours whole / arm kernel) | - | 1435.9 | 220.9 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 896.2 | 896.2..896.2 | 1 | 0.272 | - | - | 896.2 | - (stored kernel) | 0.272 (MIXED ours whole / arm kernel) | - | 1487.0 | 220.9 | accuracy=0.953559 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 687.6 ms) = 0.355; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 550.3 ms) = 0.443. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 243.9 | 243.9..243.9 | 1 | - | - | 243.9 | - | - (stored whole) | - | - | - | - | accuracy=0.862684, logloss=0.313008 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 604.6 | 604.6..604.6 | 1 | 0.403 | - | - | 604.6 | - (stored kernel) | 0.403 (MIXED ours whole / arm kernel) | - | 1098.8 | 404.2 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 550.3 | 550.3..550.3 | 1 | 0.443 | - | - | 550.3 | - (stored kernel) | 0.443 (MIXED ours whole / arm kernel) | - | 1150.4 | 404.2 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 630.5 | 630.5..630.5 | 1 | 0.387 | - | - | 630.5 | - (stored kernel) | 0.387 (MIXED ours whole / arm kernel) | - | 1099.3 | 404.2 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 725.4 | 725.4..725.4 | 1 | 0.336 | - | - | 725.4 | - (stored kernel) | 0.336 (MIXED ours whole / arm kernel) | - | 1150.3 | 404.2 | accuracy=0.868056 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 722.8 | 722.8..722.8 | 1 | 0.337 | - | - | 722.8 | - (stored kernel) | 0.337 (MIXED ours whole / arm kernel) | - | 1436.4 | 220.9 | accuracy=0.868001 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 687.6 | 687.6..687.6 | 1 | 0.355 | - | - | 687.6 | - (stored kernel) | 0.355 (MIXED ours whole / arm kernel) | - | 1487.1 | 220.9 | accuracy=0.868001 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 653.6 ms) = 0.371; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 456.3 ms) = 0.531. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 242.4 | 242.4..242.4 | 1 | - | - | 242.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.978602, rmse=0.169476 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 456.3 | 456.3..456.3 | 1 | 0.531 | - | - | 456.3 | - (stored kernel) | 0.531 (MIXED ours whole / arm kernel) | - | 1114.9 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 545.8 | 545.8..545.8 | 1 | 0.444 | - | - | 545.8 | - (stored kernel) | 0.444 (MIXED ours whole / arm kernel) | - | 1165.1 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 555.1 | 555.1..555.1 | 1 | 0.437 | - | - | 555.1 | - (stored kernel) | 0.437 (MIXED ours whole / arm kernel) | - | 1109.3 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 517.3 | 517.3..517.3 | 1 | 0.469 | - | - | 517.3 | - (stored kernel) | 0.469 (MIXED ours whole / arm kernel) | - | 1161.1 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 653.6 | 653.6..653.6 | 1 | 0.371 | - | - | 653.6 | - (stored kernel) | 0.371 (MIXED ours whole / arm kernel) | - | 1480.5 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 680.2 | 680.2..680.2 | 1 | 0.356 | - | - | 680.2 | - (stored kernel) | 0.356 (MIXED ours whole / arm kernel) | - | 1531.6 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 648.1 ms) = 0.374; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 570.0 ms) = 0.425. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 242.4 | 242.4..242.4 | 1 | - | - | 242.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.743904, rmse=0.548825 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 570.0 | 570.0..570.0 | 1 | 0.425 | - | - | 570.0 | - (stored kernel) | 0.425 (MIXED ours whole / arm kernel) | - | 1114.6 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 634.2 | 634.2..634.2 | 1 | 0.382 | - | - | 634.2 | - (stored kernel) | 0.382 (MIXED ours whole / arm kernel) | - | 1164.9 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 598.8 | 598.8..598.8 | 1 | 0.405 | - | - | 598.8 | - (stored kernel) | 0.405 (MIXED ours whole / arm kernel) | - | 1110.5 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 528.2 | 528.2..528.2 | 1 | 0.459 | - | - | 528.2 | - (stored kernel) | 0.459 (MIXED ours whole / arm kernel) | - | 1161.4 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 745.9 | 745.9..745.9 | 1 | 0.325 | - | - | 745.9 | - (stored kernel) | 0.325 (MIXED ours whole / arm kernel) | - | 1480.5 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 648.1 | 648.1..648.1 | 1 | 0.374 | - | - | 648.1 | - (stored kernel) | 0.374 (MIXED ours whole / arm kernel) | - | 1531.8 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 232.2 | 232.2..232.2 | 1 | - | - | 232.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 6394.7 | 6394.7..6394.7 | 1 | 0.036 | - | 6394.7 | 6394.7 | 0.00 (cpu-arm) | 0.036 (whole/whole) | - | 1455.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 690.6 | 690.6..690.6 | 1 | 0.336 | - | 1472.1 | 690.6 | 781.50 (upload_ms_untimed) | 0.158 (whole/whole) | - | 3005.5 | 1442.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x220, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 10.0 | 10.0..10.0 | 1 | - | - | 10.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 184.7 | 184.7..184.7 | 1 | 0.054 | - | 184.7 | 184.7 | 0.00 (cpu-arm) | 0.054 (whole/whole) | - | 265.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 64.9 | 64.9..64.9 | 1 | 0.154 | - | 124.6 | 64.9 | 59.75 (upload_ms_untimed) | 0.080 (whole/whole) | - | 919.0 | 488.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x11, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0606)`, ran on NVIDIA L40S (nv, RunPod) job n0606

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 1588.5 | 1588.5..1588.5 | 1 | - | - | 1588.5 | - | - (stored whole) | - | - | - | - | accuracy=0.920330 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0606 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 36456.8 | 36456.8..36456.8 | 1 | 0.044 | - | 36456.8 | 36456.8 | 0.00 (cpu-arm) | 0.044 (whole/whole) | - | 1138.6 | - | accuracy=0.910200 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 4933.9 | 4933.9..4933.9 | 1 | 0.322 | - | 5659.6 | 4933.9 | 725.75 (upload_ms_untimed) | 0.281 (whole/whole) | - | 3014.7 | 1374.0 | accuracy=0.809350 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0606)`, ran on NVIDIA L40S (nv, RunPod) job n0606

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 1036.6 | 1036.6..1036.6 | 1 | - | - | 1036.6 | - | - (stored whole) | - | - | - | - | accuracy=0.755330 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0606 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 14132.8 | 14132.8..14132.8 | 1 | 0.073 | - | 14132.8 | 14132.8 | 0.00 (cpu-arm) | 0.073 (whole/whole) | - | 263.2 | - | accuracy=0.752520 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1642.8 | 1642.8..1642.8 | 1 | 0.631 | - | 1710.5 | 1642.8 | 67.73 (upload_ms_untimed) | 0.606 (whole/whole) | - | 1087.7 | 498.0 | accuracy=0.703120 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 1561.5 | 1561.5..1561.5 | 1 | - | - | 1561.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.327829, rmse=0.684858 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 55283.2 | 55283.2..55283.2 | 1 | 0.028 | - | 55283.2 | 55283.2 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 1133.1 | - | finite=True, r2=-2.197e+24, rmse=1.238e+12 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 4959.8 | 4959.8..4959.8 | 1 | 0.315 | - | 5681.3 | 4959.8 | 721.44 (upload_ms_untimed) | 0.275 (whole/whole) | - | 2924.6 | 1374.0 | finite=True, r2=0.327768, rmse=0.684889 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 1024.3 | 1024.3..1024.3 | 1 | - | - | 1024.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908969, rmse=4.805411 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 13336.4 | 13336.4..13336.4 | 1 | 0.077 | - | 13336.4 | 13336.4 | 0.00 (cpu-arm) | 0.077 (whole/whole) | - | 259.8 | - | finite=True, r2=0.880681, rmse=5.501638 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1678.8 | 1678.8..1678.8 | 1 | 0.610 | - | 1731.5 | 1678.8 | 52.64 (upload_ms_untimed) | 0.592 (whole/whole) | - | 968.0 | 498.0 | finite=True, r2=0.908979, rmse=4.805168 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### simple-imputer / istella (rows full, shape X 1000000x220; X_true 1000000x220; Xq 100000x220; Xq_true 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 240.2 | 240.2..240.2 | 1 | - | - | 240.2 | - | - (stored whole) | - | - | - | - | masked_rmse=346849.129968 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 36684.7 | 36684.7..36684.7 | 1 | 0.007 | - | 36684.7 | 36684.7 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 8060.1 | - | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 252.8 | 252.8..252.8 | 1 | 0.950 | - | 928.4 | 252.8 | 675.56 (upload_ms_untimed) | 0.259 (whole/whole) | - | 4854.0 | 1442.0 | masked_rmse=346849.129968, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### simple-imputer / taxi (rows full, shape X 1000000x11; X_true 1000000x11; Xq 100000x11; Xq_true 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 9.8 | 9.8..9.8 | 1 | - | - | 9.8 | - | - (stored whole) | - | - | - | - | masked_rmse=5.985180 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1480.4 | 1480.4..1480.4 | 1 | 0.007 | - | 1480.4 | 1480.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 597.8 | - | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 18.1 | 18.1..18.1 | 1 | 0.542 | - | 73.6 | 18.1 | 55.41 (upload_ms_untimed) | 0.134 (whole/whole) | - | 1020.8 | 488.0 | masked_rmse=5.985180, max_abs_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 54.2 | 54.2..54.2 | 1 | - | - | 54.2 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=1.883381 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 32.5 | 32.5..32.5 | 1 | 1.671 | - | 32.5 | 32.5 | 0.00 (cpu-arm) | 1.671 (whole/whole) | - | 1134.7 | - | mean_abs_distortion=0.474347 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 5.9 | 5.9..5.9 | 1 | 9.213 | - | 748.3 | 5.9 | 742.40 (upload_ms_untimed) | 0.072 (whole/whole) | - | 2991.5 | 1274.0 | mean_abs_distortion=0.876709 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 4.7 | 4.7..4.7 | 1 | - | - | 4.7 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.147163 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.9 | 3.9..3.9 | 1 | 1.186 | - | 3.9 | 3.9 | 0.00 (cpu-arm) | 1.186 (whole/whole) | - | 257.7 | - | mean_abs_distortion=0.381016 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 4.3 | 4.3..4.3 | 1 | 1.090 | - | 63.6 | 4.3 | 59.32 (upload_ms_untimed) | 0.073 (whole/whole) | - | 973.8 | 478.0 | mean_abs_distortion=0.266849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 42.9 | 42.9..42.9 | 1 | - | - | 42.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 641.8 | 641.8..641.8 | 1 | 0.067 | - | 641.8 | 641.8 | 0.00 (cpu-arm) | 0.067 (whole/whole) | - | 3265.2 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 417.4 | 417.4..417.4 | 1 | 0.103 | - | 1108.0 | 417.4 | 690.57 (upload_ms_untimed) | 0.039 (whole/whole) | - | 2990.1 | 1442.0 | max_abs_diff_vs_sklearn=0.022049, output_shape=100000x220, rel_diff_vs_sklearn=0.0001523 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 91.4 | 91.4..91.4 | 1 | 0.033 | - | 91.4 | 91.4 | 0.00 (cpu-arm) | 0.033 (whole/whole) | - | 358.3 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 53.8 | 53.8..53.8 | 1 | 0.057 | - | 110.9 | 53.8 | 57.01 (upload_ms_untimed) | 0.027 (whole/whole) | - | 874.1 | 488.0 | max_abs_diff_vs_sklearn=0.0003331, output_shape=100000x11, rel_diff_vs_sklearn=3.216e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 958.4 | 958.4..958.4 | 1 | - | - | 958.4 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=662.224271, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 13448.4 | 13448.4..13448.4 | 1 | 0.071 | - | 13448.4 | 13448.4 | 0.00 (cpu-arm) | 0.071 (whole/whole) | - | 8679.1 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 267.8 | 267.8..267.8 | 1 | 3.579 | - | - | 267.8 | - (stored kernel) | 3.579 (MIXED ours whole / arm kernel) | - | 1859.6 | 5176.3 | max_rel_singular_value_error=7.758526, relative_reconstruction_error_100k_rows=2.861e-05 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 242.8 | 242.8..242.8 | 1 | 3.947 | - | - | 242.8 | - (stored kernel) | 3.947 (MIXED ours whole / arm kernel) | - | 2646.8 | 9146.0 | max_rel_singular_value_error=13326.362332, relative_reconstruction_error_100k_rows=2.385e-06 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0088)`, ran on NVIDIA L40S (nv2, RunPod) job v0088

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a1c96a5ba) | identical | 29.6 | 29.6..29.6 | 1 | - | - | 29.6 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | - | ok (main@a1c96a5ba nv2/v0088 2026-10-11; identity vs amd-mi325x: MATCH) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 319.0 | 319.0..319.0 | 1 | 0.093 | - | 319.0 | 319.0 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 450.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host f003873fb257) | torch 2.13.0+cu129 | opponent | 8.0 | 8.0..8.0 | 1 | 3.690 | - | - | 8.0 | - (stored kernel) | 3.690 (MIXED ours whole / arm kernel) | - | 826.4 | 1186.8 | max_rel_singular_value_error=2.595e-06, relative_reconstruction_error_100k_rows=7.244e-06 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host f003873fb257) | cupy 14.2.0 | opponent | 8.0 | 8.0..8.0 | 1 | 3.698 | - | - | 8.0 | - (stored kernel) | 3.698 (MIXED ours whole / arm kernel) | - | 640.7 | 2766.0 | max_rel_singular_value_error=3.763e-06, relative_reconstruction_error_100k_rows=6.886e-06 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0073)`, ran on NVIDIA L40S (nv2, RunPod) job v0073

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 207.0 | 207.0..207.0 | 1 | - | - | 207.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@c69768593 nv2/v0073 2026-10-11; identity vs amd-mi325x: MATCH) |
| gpytorch-gpu | gpytorch | gpu | NVIDIA L40S (host f003873fb257) | gpytorch 1.15.2 | opponent | 33.4 | 33.4..33.4 | 1 | 6.201 | - | - | 33.4 | - (stored kernel) | 6.201 (MIXED ours whole / arm kernel) | - | 2137.2 | 1492.0 | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | gpytorch 1.15.2 | opponent | 14121.1 | 14121.1..14121.1 | 1 | 0.015 | - | 14121.1 | 14121.1 | 0.00 (cpu-arm) | 0.015 (whole/whole) | - | 3161.9 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0073)`, ran on NVIDIA L40S (nv2, RunPod) job v0073

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c69768593) | identical | 332.8 | 332.8..332.8 | 1 | - | - | 332.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.194982, rmse=17.723482 | - | main board, one scored run | - | ok (main@c69768593 nv2/v0073 2026-10-11; identity vs amd-mi325x: MATCH) |
| gpytorch-gpu | gpytorch | gpu | NVIDIA L40S (host f003873fb257) | gpytorch 1.15.2 | opponent | 36.3 | 36.3..36.3 | 1 | 9.171 | - | - | 36.3 | - (stored kernel) | 9.171 (MIXED ours whole / arm kernel) | - | 1158.6 | 1395.0 | finite=True, r2=-0.209527, rmse=17.831018 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | gpytorch 1.15.2 | opponent | 10845.8 | 10845.8..10845.8 | 1 | 0.031 | - | 10845.8 | 10845.8 | 0.00 (cpu-arm) | 0.031 (whole/whole) | - | 2183.1 | - | finite=True, r2=-0.209546, rmse=17.831162 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0029)`, ran on NVIDIA L40S (nv2, RunPod) job v0029

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 127.9 | 127.9..127.9 | 1 | - | - | 127.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0029 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 542.6 | 542.6..542.6 | 1 | 0.236 | - | 542.6 | 542.6 | 0.00 (cpu-arm) | 0.236 (whole/whole) | - | 340.2 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 413.6 | 413.6..413.6 | 1 | 0.309 | - | 441.2 | 413.6 | 27.59 (upload_ms_untimed) | 0.290 (whole/whole) | - | 1086.5 | 684.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x8, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0029)`, ran on NVIDIA L40S (nv2, RunPod) job v0029

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@52d438ade) | identical | 99.3 | 99.3..99.3 | 1 | - | - | 99.3 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@52d438ade nv2/v0029 2026-10-11; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 227.5 | 227.5..227.5 | 1 | 0.436 | - | 227.5 | 227.5 | 0.00 (cpu-arm) | 0.436 (whole/whole) | - | 296.9 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 1182.4 | 1182.4..1182.4 | 1 | 0.084 | - | 1209.4 | 1182.4 | 26.97 (upload_ms_untimed) | 0.082 (whole/whole) | - | 1330.0 | 614.0 | max_abs_diff_vs_sklearn=0.000000, output_shape=100000x5, rel_diff_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 55.7 | 55.7..55.7 | 1 | - | - | 55.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436610 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 284.0 | 284.0..284.0 | 1 | 0.196 | - | 284.0 | 284.0 | 0.00 (cpu-arm) | 0.196 (whole/whole) | - | 467.4 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 82.5 | 82.5..82.5 | 1 | 0.676 | - | 82.5 | 82.5 | 0.00 (cpu-arm) | 0.676 (whole/whole) | - | 58.3 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0097)`, ran on NVIDIA L40S (nv2, RunPod) job v0097

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 170.7 | 170.7..170.7 | 1 | - | - | 170.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.020604 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0097 2026-10-11; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1274.4 | 1274.4..1274.4 | 1 | 0.134 | - | 1274.4 | 1274.4 | 0.00 (cpu-arm) | 0.134 (whole/whole) | - | 467.3 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 102.3 | 102.3..102.3 | 1 | 1.669 | - | 102.3 | 102.3 | 0.00 (cpu-arm) | 1.669 (whole/whole) | - | 58.1 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / istella (rows full, shape X 100000x220; Xq 10000x220; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0530)`, ran on NVIDIA L40S (nv, RunPod) job n0530

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 17.4 | 17.4..17.4 | 1 | - | - | 17.4 | - | - (stored whole) | - | - | - | - | max_additivity_error=1.175e-06 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0530 2026-10-08; identity vs amd-mi325x: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 822.3 | 822.3..822.3 | 1 | 0.021 | - | 822.3 | 822.3 | 0.00 (cpu-arm) | 0.021 (whole/whole) | - | 1561.9 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | 803.0 | 803.0..803.0 | 1 | 0.022 | - | 803.0 | 803.0 | 0.00 (cpu-arm) | 0.022 (whole/whole) | - | 1439.3 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 44.3 | 44.3..44.3 | 1 | 0.392 | - | 44.3 | - | - (stored whole) | 0.392 (whole/whole) | - | 1526.3 | 530.0 | max_additivity_error=1.956e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | 572.6 | 572.6..572.6 | 1 | 0.030 | - | 572.6 | 572.6 | 0.00 (cpu-arm) | 0.030 (whole/whole) | - | 1526.3 | - | max_additivity_error=4.441e-15 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / taxi (rows full, shape X 100000x11; Xq 10000x11; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0530)`, ran on NVIDIA L40S (nv, RunPod) job n0530

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 2.2 | 2.2..2.2 | 1 | - | - | 2.2 | - | - (stored whole) | - | - | - | - | max_additivity_error=3.858e-05 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0530 2026-10-08; identity vs amd-mi325x: MATCH) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 357.4 | 357.4..357.4 | 1 | 0.006 | - | 357.4 | 357.4 | 0.00 (cpu-arm) | 0.006 (whole/whole) | - | 485.3 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | 338.8 | 338.8..338.8 | 1 | 0.006 | - | 338.8 | 338.8 | 0.00 (cpu-arm) | 0.006 (whole/whole) | - | 374.0 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host f003873fb257) | xgboost 3.2.0 | opponent | 15.1 | 15.1..15.1 | 1 | 0.143 | - | 15.1 | - | - (stored whole) | 0.143 (whole/whole) | - | 449.1 | 500.0 | max_additivity_error=5.402e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | 234.4 | 234.4..234.4 | 1 | 0.009 | - | 234.4 | 234.4 | 0.00 (cpu-arm) | 0.009 (whole/whole) | - | 263.0 | - | max_additivity_error=5.684e-13 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0128)`, ran on NVIDIA L40S (nv2, RunPod) job v0128

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1949.7 | 1949.7..1949.7 | 1 | - | - | 1949.7 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.992124 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0128 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 43202.0 | 43202.0..43202.0 | 1 | 0.045 | - | 43202.0 | 43202.0 | 0.00 (cpu-arm) | 0.045 (whole/whole) | - | 330.8 | - | trustworthiness_k15=0.992182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 454.0 | 454.0..454.0 | 1 | 4.294 | - | 487.3 | 454.0 | 33.26 (upload_ms_untimed) | 4.001 (whole/whole) | - | 921.1 | 464.0 | trustworthiness_k15=0.990033 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/round-20261011/ingest2/nv2-results.txt (nv2/v0129)`, ran on NVIDIA L40S (nv2, RunPod) job v0129

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@c8ef6c66d) | identical | 1070.1 | 1070.1..1070.1 | 1 | - | - | 1070.1 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.998921 | - | main board, one scored run | - | ok (main@c8ef6c66d nv2/v0129 2026-10-11; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 38518.3 | 38518.3..38518.3 | 1 | 0.028 | - | 38518.3 | 38518.3 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 364.1 | - | trustworthiness_k15=0.998823 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host f003873fb257) | cuml 26.08.00 | opponent | 397.5 | 397.5..397.5 | 1 | 2.692 | - | 406.2 | 397.5 | 8.75 (upload_ms_untimed) | 2.634 (whole/whole) | - | 900.1 | 448.0 | trustworthiness_k15=0.998353 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML GaussianMixture, GaussianProcessRegressor/Classifier, Nystroem, RBFSampler: cuML 26.8.0 has none; scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: faiss-gpu: no pinned PyPI wheel for this image; cuVS ivf_flat from the pinned rapids set (cuvs-cu12==26.8.1) is the IVF-Flat GPU arm.
- Classical, wave 2, not planned on this vendor: umap-learn and faiss-cpu: not installed on NVIDIA; cuML UMAP and cuVS are the arms.
- Classical, wave 2, not planned on this vendor: cuML SpectralClustering/SpectralEmbedding: present only in newer cuML; if the pinned 26.8.0 lacks them the arm refuses by name and scikit-learn stands beside it.
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
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: mamba1-decode, mamba2-decode, mamba3-decode and samba-decode race ours alone: the repo's torch Mamba references are full-sequence scans with no carried-state decode step (a torch decode twin is not written yet)
- Neural, not planned on this vendor: torch-compile-* on transformer-decode: the twin is a per-token loop over a growing KV cache; transformer-decode races the eager arms only

