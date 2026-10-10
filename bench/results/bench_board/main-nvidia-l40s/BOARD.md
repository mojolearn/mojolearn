# mojolearn benchmark board

Generated 2026-10-10T23:00:40Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD nvidia-l40s, version label main@ca25d9321. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 191 races; oldest cell main@89485bc96 (2026-10-08T13:27:06Z), newest cell main@ca25d9321 (2026-10-10T19:37:56Z). Boxes: NVIDIA L40S (nv, RunPod), NVIDIA L40S (nv2, RunPod).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 232 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: one scored run per cell (lq RACE ALGOS lines, lq CMD bench_board summaries); the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 80, n/a 111).

> Opponents: copied from the stored opponent boards (opponents-default-20261006, opponents-specific-20261006, release-board-resume-r2), never re-run here; `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 2 races.

> Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax. A cell of ours whose output hash equals a copied opponent's shows quality identical_to=<arm> (the same bits) where the own-host reference gave none.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 80, n/a 111.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 5 failed runs.

| lane | dataset | commit | hardware | version | box/job | reason | ok cell on the board |
|---|---|---|---|---|---|---|---|
| conv2d | synthetic | main@e5f3f2ed8 | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@e5f3f2ed8) | nv/n0570 | NO-RECORD | none |
| moe | synthetic | main@e5f3f2ed8 | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@e5f3f2ed8) | nv/n0570 | NO-RECORD | none |
| resnet-block | synthetic | main@e5f3f2ed8 | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@e5f3f2ed8) | nv/n0570 | NO-RECORD | none |
| gbdt-categorical | istella | main@de2b2b739 | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | nv/n0320 | NO-RECORD | none |
| gbdt-categorical | taxi | main@de2b2b739 | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | nv/n0320 | REFUSED(Exception_during_warm-up:_At_max/mojo/max/gpu/host/device_context.mojo:4073:35:_CUDA_call_failed:_CUDA_ERROR_OUT_OF_MEMO) | none |

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

Races: 191 planned, 191 done, 0 failed, 0 unsupported, 0 pending. Cells: 732 (REFUSED 25, ok 707).

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
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002792; torch-eager-tf32 0.000000; torch-compile-tf32 0.002792; torch-eager-bf16 0.000000; torch-compile-bf16 0.002792 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.109e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.109e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.109e-08 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001863; torch-eager-tf32 0.000000; torch-compile-tf32 0.001863; torch-eager-bf16 0.000000; torch-compile-bf16 0.001863 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 4.327e-08; torch-eager-tf32 0.000000; torch-compile-tf32 4.327e-08; torch-eager-bf16 0.000000; torch-compile-bf16 4.327e-08 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | - | 0.794050 | cuml-gpu 0.794050; sklearn-cpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | - | 5.350625 | cuml-gpu 5.350631; sklearn-cpu 4.278741 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | 0.755560 | cuml-gpu 0.755560; sklearn-cpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | 0.557803 | cuml-gpu 0.557803; sklearn-cpu 0.557802 |
| algos | categorical-nb | istella | accuracy (higher is better) | - | 0.838850 | cuml-gpu 0.838850; sklearn-cpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | - | 0.412625 | cuml-gpu 0.412625; sklearn-cpu 0.412625 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | 0.765850 | cuml-gpu 0.765850; sklearn-cpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | 0.538866 | cuml-gpu 0.538866; sklearn-cpu 0.538866 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | torch-gpu 1.509e-07; cupy-gpu 1.365e-07; numpy-cpu 3.928e-08 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | - | 1.000000 | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000; torch-eager-tf32 1.000000; torch-compile-tf32 1.000000; torch-eager-bf16 1.000000; torch-compile-bf16 1.000000 |
| algos | complement-nb | istella | accuracy (higher is better) | - | 0.849360 | cuml-gpu 0.849350; sklearn-cpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | - | 3.762524 | cuml-gpu 3.763060; sklearn-cpu 3.174763 |
| algos | complement-nb | taxi | accuracy (higher is better) | - | 0.678020 | cuml-gpu 0.678060; sklearn-cpu 0.678030 |
| algos | complement-nb | taxi | logloss (lower is better) | - | 0.715492 | cuml-gpu 0.715531; sklearn-cpu 0.715493 |
| algos | complement-nb | text | accuracy (higher is better) | - | 0.983067 | cuml-gpu 0.983067; sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | 0.559491 | cuml-gpu 0.559490; sklearn-cpu 0.557285 |
| algos | connected-components | istella | n_components | - | 81 | cugraph-gpu 81; networkx-cpu 81 |
| algos | connected-components | taxi | n_components | - | 588 | cugraph-gpu 588; networkx-cpu 588 |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 3417.849541; torch-compile-bf16 3524.661064 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.003296; torch-compile-bf16 0.003294 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.506768; torch-eager-tf32 382.931903; torch-compile-tf32 382.931903; torch-eager-bf16 3418.337554; torch-compile-bf16 3433.596343 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.007e-07; torch-eager-tf32 0.0003021; torch-compile-tf32 0.0003021; torch-eager-bf16 0.003382; torch-compile-bf16 0.003380 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | - | 13.931153 | statsmodels-cpu 26.588738; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | - | 96.690449 | statsmodels-cpu 196.955273; statsforecast-cpu 96.685568 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 5.95e-05 | torch-gpu 1.044e-06; cupy-gpu 1.044e-06; numpy-cpu 3.49e-08 |
| algos | eigh | synthetic | relative_residual | - | 5.251e-05 | torch-gpu 1.016e-06; cupy-gpu 1.016e-06; numpy-cpu 2.824e-08 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | enet-cv | istella | r2 (higher is better) | - | 0.326805 | sklearn-cpu 0.326805 |
| algos | enet-cv | istella | rmse (lower is better) | - | 0.685379 | sklearn-cpu 0.685379 |
| algos | enet-cv | taxi | r2 (higher is better) | - | 0.909004 | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | - | 4.804486 | sklearn-cpu 4.804486 |
| algos | gaussian-nb | istella | accuracy (higher is better) | - | 0.876570 | cuml-gpu 0.876570; sklearn-cpu 0.876530 |
| algos | gaussian-nb | istella | logloss (lower is better) | - | 3.574420 | cuml-gpu 3.416741; sklearn-cpu 3.417392 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | 0.719820 | cuml-gpu 0.719810; sklearn-cpu 0.719900 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | 1.132249 | cuml-gpu 1.132317; sklearn-cpu 1.133898 |
| algos | gaussian-rp | istella | mean_abs_distortion | - | 0.680693 | cuml-gpu 0.443920; sklearn-cpu 0.177966 |
| algos | gaussian-rp | taxi | mean_abs_distortion | - | 0.345752 | cuml-gpu 0.302259; sklearn-cpu 0.339791 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.009646; torch-eager-tf32 281.122146; torch-compile-tf32 281.117866; torch-eager-bf16 2718.059111; torch-compile-bf16 2718.055850 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.041e-07; torch-eager-tf32 0.0002661; torch-compile-tf32 0.0002661; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002327; torch-eager-tf32 0.001863; torch-compile-tf32 0.002794; torch-eager-bf16 1509.509282; torch-compile-bf16 1509.508234 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.803e-08; torch-eager-tf32 7.223e-08; torch-compile-tf32 9.801e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.009731; torch-eager-tf32 0.000000; torch-compile-tf32 0.009731; torch-eager-bf16 0.000000; torch-compile-bf16 0.009731 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.64e-08; torch-eager-tf32 0.000000; torch-compile-tf32 9.64e-08; torch-eager-bf16 0.000000; torch-compile-bf16 9.64e-08 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.134110; torch-eager-tf32 430.934131; torch-compile-tf32 430.934131; torch-eager-bf16 3907.114267; torch-compile-bf16 3678.210080 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.7e-08; torch-eager-tf32 0.0002969; torch-compile-tf32 0.0002969; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.238419; torch-eager-tf32 0.029851; torch-compile-tf32 1475.334167; torch-eager-bf16 5460.333333; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.954e-08; torch-eager-tf32 4.218e-08; torch-compile-tf32 0.0002145; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | 0.971625 | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-tf32 0.971842; torch-compile-tf32 0.971842; torch-eager-bf16 0.971951; torch-compile-bf16 0.971951 |
| algos | gru-clf | synthetic | logloss (lower is better) | - | 0.070054 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | 0.865723 | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-tf32 0.865668; torch-compile-tf32 0.865668; torch-eager-bf16 0.865668; torch-compile-bf16 0.865668 |
| algos | gru-clf | taxi-hourly | logloss (lower is better) | - | 0.303537 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | gru-reg | synthetic | r2 (higher is better) | - | 0.982337 | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-tf32 0.981946; torch-compile-tf32 0.981946; torch-eager-bf16 0.981898; torch-compile-bf16 0.981898 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | 0.153975 | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-tf32 0.155672; torch-compile-tf32 0.155672; torch-eager-bf16 0.155878; torch-compile-bf16 0.155878 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | 0.747875 | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-tf32 0.748219; torch-compile-tf32 0.748219; torch-eager-bf16 0.748325; torch-compile-bf16 0.748325 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | 0.544554 | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-tf32 0.544182; torch-compile-tf32 0.544182; torch-eager-bf16 0.544067; torch-compile-bf16 0.544067 |
| algos | incremental-pca | istella | explained_variance_fraction | - | 1.000000 | cuml-gpu 1.000000; sklearn-cpu 1.000000 |
| algos | incremental-pca | taxi | explained_variance_fraction | - | 0.999995 | cuml-gpu 0.999995; sklearn-cpu 0.999995 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | 0.802100 | cuvs-gpu 0.791300; faiss-cpu - |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.981250 | cuvs-gpu 0.976875; faiss-cpu 0.979450 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | - | 0.809175 | cuvs-gpu 0.993625; faiss-cpu - |
| algos | ivf-refine | taxi | recall_at_10 (higher is better) | - | 0.999675 | cuvs-gpu 0.999050; faiss-cpu 0.999375 |
| algos | ivf-sq | taxi | recall_at_10 (higher is better) | - | 0.934975 | cuvs-gpu 0.772500; faiss-cpu 0.857050 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | 8.307e-08 | cuml-gpu 0.032177; shap-cpu 8.574e-15 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | 1.056e-07 | cuml-gpu 7.647e-07; shap-cpu 8.154e-15 |
| algos | knn-imputer | istella | masked_rmse | - | 323953.237332 | sklearn-cpu 986208.700423 |
| algos | knn-imputer | taxi | masked_rmse | - | 6.151696 | sklearn-cpu 5.263919 |
| algos | lars | istella | r2 (higher is better) | - | 0.309043 | cuml-gpu 0.328088; sklearn-cpu -4.245e+13 |
| algos | lars | istella | rmse (lower is better) | - | 0.694362 | cuml-gpu 0.684726; sklearn-cpu 5.442e+06 |
| algos | lars | taxi | r2 (higher is better) | - | 0.908981 | cuml-gpu 0.908983; sklearn-cpu 0.908983 |
| algos | lars | taxi | rmse (lower is better) | - | 4.805109 | cuml-gpu 4.805052; sklearn-cpu 4.805055 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.325506 | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.686040 | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909038 | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803593 | sklearn-cpu 4.803593 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.006419; torch-eager-tf32 0.000000; torch-compile-tf32 0.006419; torch-eager-bf16 0.000000; torch-compile-bf16 0.006419 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.316e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.316e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.316e-08 |
| algos | lda-clf | istella | accuracy (higher is better) | - | 0.912830 | sklearn-cpu 0.899510 |
| algos | lda-clf | istella | logloss (lower is better) | - | 0.235875 | sklearn-cpu 0.438978 |
| algos | lda-clf | taxi | accuracy (higher is better) | - | 0.762530 | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | - | 0.539763 | sklearn-cpu 0.539767 |
| algos | louvain | istella | modularity | - | 0.911187 | cugraph-gpu 0.909430; networkx-cpu 0.908460 |
| algos | louvain | istella | n_communities | - | 40 | cugraph-gpu 41; networkx-cpu 40 |
| algos | louvain | taxi | modularity | - | 0.941953 | cugraph-gpu 0.941795; networkx-cpu 0.940781 |
| algos | louvain | taxi | n_communities | - | 58 | cugraph-gpu 62; networkx-cpu 56 |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | 0.967068 | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-tf32 0.968696; torch-compile-tf32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-clf | synthetic | logloss (lower is better) | - | 0.079720 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | 0.870443 | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-tf32 0.868164; torch-compile-tf32 0.868164; torch-eager-bf16 0.868327; torch-compile-bf16 0.868327 |
| algos | lstm-clf | taxi-hourly | logloss (lower is better) | - | 0.297146 | torch-eager-fp32 -; torch-compile-fp32 -; torch-eager-tf32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 - |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | 0.979398 | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-tf32 0.981013; torch-compile-tf32 0.981013; torch-eager-bf16 0.980998; torch-compile-bf16 0.980998 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | 0.166295 | torch-eager-fp32 0.159641; torch-compile-fp32 0.159641; torch-eager-tf32 0.159642; torch-compile-tf32 0.159642; torch-eager-bf16 0.159706; torch-compile-bf16 0.159706 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | 0.754306 | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-tf32 0.751680; torch-compile-tf32 0.751680; torch-eager-bf16 0.751712; torch-compile-bf16 0.751712 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | 0.537563 | torch-eager-fp32 0.540430; torch-compile-fp32 0.540430; torch-eager-tf32 0.540429; torch-compile-tf32 0.540429; torch-eager-bf16 0.540394; torch-compile-bf16 0.540394 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | torch-gpu nan; cupy-gpu 0.849957; numpy-cpu 0.876581 |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | torch-gpu 0.756366; cupy-gpu 0.756366; numpy-cpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 3.386e-07; cupy-gpu 3.386e-07; scipy-cpu 4.275e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 3.386e-07; cupy-gpu 3.386e-07; numpy-cpu 3.259e-08 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.026193; torch-eager-tf32 1004.691291; torch-compile-tf32 1004.691291; torch-eager-bf16 22834.612745; torch-compile-bf16 22851.987745 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.548e-07; torch-eager-tf32 0.017534; torch-compile-tf32 0.017534; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | multinomial-nb | istella | accuracy (higher is better) | - | 0.853620 | cuml-gpu 0.853620; sklearn-cpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | - | 3.628565 | cuml-gpu 3.628599; sklearn-cpu 3.087499 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | - | 0.723160 | cuml-gpu 0.723160; sklearn-cpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | - | 0.590725 | cuml-gpu 0.590750; sklearn-cpu 0.590725 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | cuml-gpu 0.983067; sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | cuml-gpu 0.559524; sklearn-cpu 0.557319 |
| algos | nadam | synthetic | relative_error_vs_own_host | - | 0.000000 | torch-eager-fp32 -; torch-compile-fp32 - |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.498e-07 |
| algos | optimized-theta | synthetic | forecast_rmse (lower is better) | - | 1.438855 | statsforecast-cpu 1.437815 |
| algos | optimized-theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.150860 | statsforecast-cpu 49.356608 |
| algos | pagerank | istella | sum | - | 1.000000 | cugraph-gpu 1.000000; networkx-cpu 1.000000 |
| algos | pagerank | taxi | sum | - | 1.000000 | cugraph-gpu 1.000000; networkx-cpu 1.000000 |
| algos | permutation-shap | istella | rel_error_vs_exact | - | 6.68e-08 | cuml-gpu 1.483e-07; shap-cpu 3.692e-10 |
| algos | permutation-shap | taxi | rel_error_vs_exact | - | 1.054e-07 | cuml-gpu 1.701e-07; shap-cpu 1.144e-15 |
| algos | qr | istella | relative_gram_difference | - | 1.485e-07 | torch-gpu 3.787e-07; cupy-gpu 3.787e-07; numpy-cpu 2.459e-08 |
| algos | qr | taxi | relative_gram_difference | - | 1.714e-07 | torch-gpu 5.971e-06; cupy-gpu 5.971e-06; numpy-cpu 3.024e-08 |
| algos | quantile | istella | r2 (higher is better) | - | -0.039998 | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | - | 0.851877 | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | - | 0.899596 | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | - | 5.046749 | sklearn-cpu 5.044706 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | 0.0002359 | torch-gpu 0.0002359; sklearn-cpu 0.0002359 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | 0.027197 | torch-gpu 0.027197; sklearn-cpu 0.027197 |
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
| algos | sgd-clf | istella | accuracy (higher is better) | - | 0.920330 | cuml-gpu 0.809350; sklearn-cpu 0.910200 |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | 0.755330 | cuml-gpu 0.703120; sklearn-cpu 0.752520 |
| algos | sgd-reg | istella | r2 (higher is better) | - | 0.327829 | cuml-gpu 0.327768; sklearn-cpu -2.197e+24 |
| algos | sgd-reg | istella | rmse (lower is better) | - | 0.684858 | cuml-gpu 0.684889; sklearn-cpu 1.238e+12 |
| algos | sgd-reg | taxi | r2 (higher is better) | - | 0.908969 | cuml-gpu 0.908979; sklearn-cpu 0.880681 |
| algos | sgd-reg | taxi | rmse (lower is better) | - | 4.805411 | cuml-gpu 4.805168; sklearn-cpu 5.501638 |
| algos | simple-imputer | istella | masked_rmse | - | 346849.129968 | cuml-gpu 346849.129968; sklearn-cpu 346849.129968 |
| algos | simple-imputer | taxi | masked_rmse | - | 5.985180 | cuml-gpu 5.985180; sklearn-cpu 5.985180 |
| algos | sparse-rp | istella | mean_abs_distortion | - | 1.883381 | cuml-gpu 0.876709; sklearn-cpu 0.474347 |
| algos | sparse-rp | taxi | mean_abs_distortion | - | 0.147163 | cuml-gpu 0.266849; sklearn-cpu 0.381016 |
| algos | svd | istella | max_rel_singular_value_error | - | 662.224271 | torch-gpu 8.068917; cupy-gpu 10270.720886; numpy-cpu 1.000000 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | 3.379e-05 | torch-gpu 2.861e-05; cupy-gpu 2.385e-06; numpy-cpu 4.1e-08 |
| algos | svd | taxi | max_rel_singular_value_error | - | 7.036e-07 | torch-gpu 2.595e-06; cupy-gpu 3.763e-06; numpy-cpu 4.308e-08 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | 1.18e-06 | torch-gpu 7.244e-06; cupy-gpu 6.886e-06; numpy-cpu 4.314e-08 |
| algos | svgp | istella | r2 (higher is better) | - | -0.106016 | gpytorch-gpu -0.106040; gpytorch-cpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | 0.878373 | gpytorch-gpu 0.878383; gpytorch-cpu 0.878383 |
| algos | theta | synthetic | forecast_rmse (lower is better) | - | 1.436610 | statsforecast-cpu 1.436557; statsmodels-cpu 1.434862 |
| algos | theta | taxi-hourly | forecast_rmse (lower is better) | - | 49.020604 | statsforecast-cpu 49.253901; statsmodels-cpu 49.311757 |
| algos | tree-shap | istella | max_additivity_error | - | 1.175e-06 | xgboost-gpu 1.956e-06; shap-cpu 1.837e-06; xgboost-cpu 1.837e-06; lightgbm-cpu 4.441e-15 |
| algos | tree-shap | taxi | max_additivity_error | - | 3.858e-05 | xgboost-gpu 5.402e-05; shap-cpu 0.0001201; xgboost-cpu 0.0001201; lightgbm-cpu 5.684e-13 |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.992124 | cuml-gpu 0.990033; sklearn-cpu 0.992182 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.998921 | cuml-gpu 0.998353; sklearn-cpu 0.998823 |
| classical | dbscan | istella | n_clusters | - | 40131 | cuml-gpu 40131 |
| classical | dbscan | istella | noise_fraction | - | 0.219391 | cuml-gpu 0.219391 |
| classical | dbscan | istella | rows | - | 1000000 | cuml-gpu 1000000 |
| classical | dbscan | taxi | n_clusters | - | 36 | cuml-gpu 36 |
| classical | dbscan | taxi | noise_fraction | - | 0.000174 | cuml-gpu 0.000174 |
| classical | dbscan | taxi | rows | - | 1000000 | cuml-gpu 1000000 |
| classical | hdbscan | istella | n_clusters | - | 47 | cuml-gpu 53 |
| classical | hdbscan | istella | noise_fraction | - | 0.253810 | cuml-gpu 0.256280 |
| classical | hdbscan | istella | rows | - | 100000 | cuml-gpu 100000 |
| classical | hdbscan | taxi | n_clusters | - | 161 | cuml-gpu 159 |
| classical | hdbscan | taxi | noise_fraction | - | 0.140550 | cuml-gpu 0.130970 |
| classical | hdbscan | taxi | rows | - | 100000 | cuml-gpu 100000 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | cuml-gpu 6.111e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | cuml-gpu -; torch-gpu - |
| classical | kmeans | istella | n_iter | - | 33 | cuml-gpu 21; torch-gpu 55 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | cuml-gpu 3.06e+08; torch-gpu 3.06e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | cuml-gpu -; torch-gpu - |
| classical | kmeans | taxi | n_iter | - | 91 | cuml-gpu 32; torch-gpu 58 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | cuml-gpu 0.976402; torch-gpu 0.981012 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | cuml-gpu 0; torch-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | cuml-gpu 0.999742; torch-gpu 0.999773 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | cuml-gpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | cuml-gpu -11031.855105; torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | cuml-gpu 87.647429; torch-gpu nan; torch-gpu-eigh 0.768590 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908836 | cuml-gpu 0.908836; torch-gpu 0.908836; torch-gpu-eigh 0.908836 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696479 | cuml-gpu 4.696488; torch-gpu 4.696480; torch-gpu-eigh 4.696490 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | cuml-gpu 1.000000; torch-gpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999997 | cuml-gpu 0.999996; torch-gpu 0.999996 |
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
| classical2 | ivf | istella | recall_at_k (higher is better) | - | 0.999925 | cuvs-gpu 0.999975 |
| classical2 | ivf | istella | rows_with_repeated_ids | - | 0 | cuvs-gpu 0 |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | 0.999650 | cuvs-gpu 0.999450 |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | 0 | cuvs-gpu 0 |
| classical2 | lasso | istella | r2 (higher is better) | - | 0.310837 | cuml-gpu 0.310837 |
| classical2 | lasso | istella | rmse (lower is better) | - | 0.693460 | cuml-gpu 0.693460 |
| classical2 | lasso | taxi | r2 (higher is better) | - | 0.908995 | cuml-gpu 0.908995 |
| classical2 | lasso | taxi | rmse (lower is better) | - | 4.804744 | cuml-gpu 4.804745 |
| classical2 | logreg | istella | accuracy (higher is better) | - | 0.924590 | cuml-gpu 0.924430 |
| classical2 | logreg | istella | logloss (lower is better) | - | 0.181249 | cuml-gpu 0.181268 |
| classical2 | logreg | istella | nonfinite_proba_rows | - | 0 | cuml-gpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | - | 0.763350 | cuml-gpu 0.763350 |
| classical2 | logreg | taxi | logloss (lower is better) | - | 0.538985 | cuml-gpu 0.538986 |
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
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-tf32 9.018733; torch-compile-fp32 9.018733; torch-compile-tf32 9.018732; torch-eager-bf16 9.018664; torch-compile-bf16 9.018669 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | torch-eager-fp32 9.018733; torch-eager-tf32 9.018732; torch-compile-fp32 9.018734; torch-compile-tf32 9.018732; torch-eager-bf16 9.018402; torch-compile-bf16 9.018669 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.418446 | torch-eager-fp32 8.418449; torch-eager-tf32 8.418548; torch-compile-fp32 8.418448; torch-compile-tf32 8.418557; torch-eager-bf16 8.417328; torch-compile-bf16 8.417006 |
| neural | lm-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-eager-bf16 -; mamba-ssm-fp32 1.192e-07; mamba-ssm-tf32 3.815e-06 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-eager-bf16 -; mamba-ssm-fp32 5.945e-08; mamba-ssm-tf32 1.902e-06 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 1.907e-06; mamba-ssm-tf32 0.0005126 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 6.485e-07; mamba-ssm-tf32 0.0001743 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 0.001235; mamba-ssm-tf32 0.001376 |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | torch-eager-fp32 -; torch-eager-tf32 -; torch-compile-fp32 -; torch-compile-tf32 -; torch-eager-bf16 -; torch-compile-bf16 -; mamba-ssm-fp32 0.0005401; mamba-ssm-tf32 0.0006018 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | torch-eager-fp32 1.160401; torch-eager-tf32 1.160392; torch-compile-fp32 1.160401; torch-compile-tf32 1.160392; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | torch-eager-fp32 1.123361; torch-eager-tf32 1.123355; torch-compile-fp32 1.123361; torch-compile-tf32 1.123355; torch-eager-bf16 1.123461; torch-compile-bf16 1.123462 |
| neural | mlp-train-step | gaussian | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | torch-eager-fp32 5.635910; torch-eager-tf32 5.635948; torch-compile-fp32 5.635910; torch-compile-tf32 5.635950; torch-eager-bf16 5.635952; torch-compile-bf16 5.635985 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | torch-eager-fp32 5.635910; torch-eager-tf32 5.635948; torch-compile-fp32 5.635910; torch-compile-tf32 5.635947; torch-eager-bf16 5.635952; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | torch-eager-fp32 4.833934; torch-eager-tf32 4.833591; torch-compile-fp32 4.833934; torch-compile-tf32 4.833560; torch-eager-bf16 4.833967; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 - |
| trees | et | istella | auc (higher is better) | - | 0.937987 | sklearn-et-cpu 0.937659 |
| trees | et | istella | logloss (lower is better) | - | 0.189989 | sklearn-et-cpu 0.190177 |
| trees | et | taxi | auc (higher is better) | - | 0.618907 | sklearn-et-cpu 0.619262 |
| trees | et | taxi | logloss (lower is better) | - | 0.526142 | sklearn-et-cpu 0.525951 |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.983304 | catboost-gpu 0.983152; xgboost-gpu 0.983622; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.156072 | catboost-gpu 0.156748; xgboost-gpu 0.149263; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.632205 | catboost-gpu 0.632335; xgboost-gpu 0.630969; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.527838 | catboost-gpu 0.527912; xgboost-gpu 0.528678; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983586 | catboost-gpu 0.983668; xgboost-gpu 0.983622; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.500000; lightgbm-cpu - |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.150066 | catboost-gpu 0.149188; xgboost-gpu 0.149263; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.356515; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.632096 | catboost-gpu 0.631766; xgboost-gpu 0.630969; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.500000 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528044 | catboost-gpu 0.528065; xgboost-gpu 0.528678; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.554692 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.907556 | catboost-gpu 0.907780; xgboost-gpu 0.910140; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.911402; lightgbm-cpu - |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.258413 | catboost-gpu 0.258149; xgboost-gpu 0.246803; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.243459; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.599380 | catboost-gpu 0.599270; xgboost-gpu 0.601200; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 0.601896 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.012595 | catboost-gpu 1.012704; xgboost-gpu 1.005204; catboost-cpu -; xgboost-cpu -; lightgbm-cpu -; lightgbm-cuda 1.003173 |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.979444 | catboost-gpu 0.979432; catboost-cpu - |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.190928 | catboost-gpu 0.190474; catboost-cpu - |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.629203 | catboost-gpu 0.628945; catboost-cpu - |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.529007 | catboost-gpu 0.528997; catboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.854545 | catboost-gpu 0.853929; xgboost-gpu 0.872796; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.719953 | catboost-gpu 0.719621; xgboost-gpu 0.738397; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.650400 | catboost-gpu 0.650025; xgboost-gpu 0.670093; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-gpu 0.980018; catboost-cpu - |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-gpu 0.187292; catboost-cpu - |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-gpu 0.630310; catboost-cpu - |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-gpu 0.528616; catboost-cpu - |
| trees | rf | istella | auc (higher is better) | - | 0.945385 | cuml-rf-gpu 0.945348 |
| trees | rf | istella | logloss (lower is better) | - | 0.182017 | cuml-rf-gpu 0.182069 |
| trees | rf | taxi | auc (higher is better) | - | 0.617838 | cuml-rf-gpu 0.617836 |
| trees | rf | taxi | logloss (lower is better) | - | 0.525953 | cuml-rf-gpu 0.525954 |


## Neural headline: ours IDENTICAL against torch bf16

What customers run is torch in bf16. The headline divides our IDENTICAL median by torch's fastest bf16 arm (eager or compile, the lower stored median); the fp32 twin (torch's fastest fp32 arm) is the second column. Per-arm ratios stay in each race's table below.

| lane | dataset | ours IDENTICAL ms | torch bf16 ms (fastest arm) | ours / torch bf16 | torch fp32 twin ms (fastest arm) | ours / torch fp32 | note |
|---|---|---|---|---|---|---|---|
| adafactor | synthetic | 13.7 | - | - | 15.3 (torch-eager-fp32) | 0.897 | no torch bf16 arm on this lane |
| adagrad | synthetic | 8.5 | - | - | 14.9 (torch-eager-fp32) | 0.570 | no torch bf16 arm on this lane |
| adamax | synthetic | 11.2 | - | - | 20.2 (torch-eager-fp32) | 0.555 | no torch bf16 arm on this lane |
| avgpool1d | synthetic | 1.0 | 1.5 (torch-eager-bf16) | 0.658 | 2.4 (torch-compile-fp32) | 0.418 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| avgpool2d | synthetic | 1.7 | 1.7 (torch-eager-bf16) | 0.996 | 1.1 (torch-compile-fp32) | 1.456 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm1d | synthetic | 1.6 | 1.8 (torch-compile-bf16) | 0.871 | 1.7 (torch-compile-fp32) | 0.941 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| batchnorm2d | synthetic | 1.2 | 1.4 (torch-eager-bf16) | 0.861 | 1.3 (torch-eager-fp32) | 0.878 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| cnn-clf | synthetic | 195.8 | 483.3 (torch-eager-bf16) | 0.405 | 458.6 (torch-eager-fp32) | 0.427 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv1d | synthetic | 10.1 | 3.3 (torch-eager-bf16) | 3.085 | 3.8 (torch-eager-fp32) | 2.644 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| conv2d | synthetic | 6.3 | 1.3 (torch-eager-bf16) | 4.722 | 2.3 (torch-eager-fp32) | 2.732 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| dropout2d | synthetic | 1.1 | - | - | 1.0 (torch-eager-fp32) | 1.130 | no torch bf16 arm on this lane |
| embedding | synthetic | 2.7 | - | - | 1.9 (torch-eager-fp32) | 1.434 | no torch bf16 arm on this lane |
| gcn | istella | 3.0 | 8.5 (torch-compile-bf16) | 0.353 | 4.5 (torch-compile-fp32) | 0.675 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gcn | taxi | 3.0 | 9.3 (torch-compile-bf16) | 0.319 | 3.3 (torch-compile-fp32) | 0.894 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-avgpool | synthetic | 2.1 | 0.6 (torch-eager-bf16) | 3.251 | 0.7 (torch-eager-fp32) | 2.820 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| global-maxpool | synthetic | 2.9 | 0.6 (torch-eager-bf16) | 5.009 | 0.6 (torch-eager-fp32) | 4.985 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | istella | 7.5 | 3.3 (torch-compile-bf16) | 2.250 | 4.2 (torch-compile-fp32) | 1.765 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| graphsage | taxi | 1.7 | 1.7 (torch-eager-bf16) | 1.022 | 1.6 (torch-eager-fp32) | 1.074 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | synthetic | 447.1 | 1311.6 (torch-compile-bf16) | 0.341 | 1103.7 (torch-eager-fp32) | 0.405 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-clf | taxi-hourly | 447.7 | 789.0 (torch-compile-bf16) | 0.567 | 591.3 (torch-eager-fp32) | 0.757 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | synthetic | 446.3 | 1333.4 (torch-compile-bf16) | 0.335 | 1133.5 (torch-eager-fp32) | 0.394 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gru-reg | taxi-hourly | 447.0 | 696.1 (torch-eager-bf16) | 0.642 | 650.5 (torch-eager-fp32) | 0.687 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lamb | synthetic | 23.8 | - | - | - | - | no torch bf16 arm on this lane |
| layernorm | synthetic | 1.1 | 1.9 (torch-eager-bf16) | 0.592 | 3.4 (torch-eager-fp32) | 0.322 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lion | synthetic | 11.0 | - | - | - | - | no torch bf16 arm on this lane |
| lstm-clf | synthetic | 546.6 | 743.9 (torch-eager-bf16) | 0.735 | 516.0 (torch-eager-fp32) | 1.059 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-clf | taxi-hourly | 546.5 | 1735.3 (torch-compile-bf16) | 0.315 | 1246.5 (torch-eager-fp32) | 0.438 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | synthetic | 545.6 | 857.8 (torch-compile-bf16) | 0.636 | 564.5 (torch-compile-fp32) | 0.966 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lstm-reg | taxi-hourly | 545.8 | 1539.5 (torch-compile-bf16) | 0.355 | 1243.7 (torch-compile-fp32) | 0.439 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool1d | synthetic | 2.2 | 1.7 (torch-compile-bf16) | 1.286 | 1.4 (torch-eager-fp32) | 1.632 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| maxpool2d | synthetic | 1.5 | 2.2 (torch-eager-bf16) | 0.654 | 1.8 (torch-eager-fp32) | 0.832 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| moe | synthetic | 14.2 | 4.3 (torch-eager-bf16) | 3.266 | 11.9 (torch-compile-fp32) | 1.189 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| nadam | synthetic | 10.8 | - | - | 20.8 (torch-eager-fp32) | 0.521 | no torch bf16 arm on this lane |
| resnet-block | synthetic | 16.2 | 2.8 (torch-eager-bf16) | 5.700 | 5.3 (torch-compile-fp32) | 3.077 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rmsprop | synthetic | 8.6 | - | - | 13.3 (torch-eager-fp32) | 0.646 | no torch bf16 arm on this lane |
| rnn-clf | synthetic | 244.0 | 873.9 (torch-compile-bf16) | 0.279 | 904.6 (torch-eager-fp32) | 0.270 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-clf | taxi-hourly | 243.9 | 642.8 (torch-compile-bf16) | 0.379 | 507.7 (torch-eager-fp32) | 0.480 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | synthetic | 242.4 | 1442.1 (torch-compile-bf16) | 0.168 | 902.4 (torch-compile-fp32) | 0.269 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| rnn-reg | taxi-hourly | 242.4 | 612.7 (torch-eager-bf16) | 0.396 | 664.7 (torch-eager-fp32) | 0.365 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| gemm | gaussian | 51.3 | 45.3 (torch-eager-bf16) | 1.133 | 49.2 (torch-compile-fp32) | 1.043 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-forward | bytes | 48.6 | 42.5 (torch-compile-bf16) | 1.144 | 45.8 (torch-compile-fp32) | 1.059 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| lm-train-step | bytes | 35.4 | 10.8 (torch-compile-bf16) | 3.290 | 21.6 (torch-compile-fp32) | 1.643 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba1-forward | gaussian | 13.0 | 193.2 (torch-eager-bf16) | 0.067 | 150.9 (torch-eager-fp32) | 0.086 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba2-forward | gaussian | 13.1 | 21.2 (torch-eager-bf16) | 0.621 | 25.9 (torch-eager-fp32) | 0.507 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mamba3-forward | gaussian | 4.1 | 16.3 (torch-compile-bf16) | 0.251 | 12.4 (torch-compile-fp32) | 0.331 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| mlp-train-step | gaussian | 2.4 | 2.0 (torch-eager-bf16) | 1.239 | 1.5 (torch-eager-fp32) | 1.658 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-forward | bytes | 5.0 | 5.5 (torch-compile-bf16) | 0.910 | 12.1 (torch-compile-fp32) | 0.413 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| samba-train-step | bytes | 39.6 | 165.9 (torch-eager-bf16) | 0.238 | 46.3 (torch-compile-fp32) | 0.854 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |
| transformer-forward | gaussian | 4.0 | 3.4 (torch-eager-bf16) | 1.165 | 3.6 (torch-eager-fp32) | 1.113 | torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax |

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

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 7232.6 | 7232.6..7232.6 | 1 | - | - | 7232.6 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 9671.9 | 9671.9..9671.9 | 1 | 0.748 | - | 9671.9 | - | - (stored whole) | 0.748 (whole/whole) | - | 5377.9 | 502.0 | auc=0.983152, logloss=0.156748 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 9340.7 | 9340.7..9340.7 | 1 | 0.774 | - | 9340.7 | - | - (stored whole) | 0.774 (whole/whole) | - | 6478.8 | 502.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 3692.4 | 3692.4..3692.4 | 1 | - | - | 3692.4 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 6393.8 | 6393.8..6393.8 | 1 | 0.577 | - | 6393.8 | - | - (stored whole) | 0.577 (whole/whole) | - | 1576.3 | 496.0 | auc=0.632335, logloss=0.527912 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host 24a11adce16e) | xgboost 3.2.0 | opponent | 3309.5 | 3309.5..3309.5 | 1 | 1.116 | - | 3309.5 | - | - (stored whole) | 1.116 (whole/whole) | - | 1670.3 | 496.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on NVIDIA L40S (nv, RunPod) job n0320

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 8941.1 | 8941.1..8941.1 | 1 | - | - | 8941.1 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 23630.6 | 23630.6..23630.6 | 1 | 0.378 | - | 23630.6 | - | - (stored whole) | 0.378 (whole/whole) | - | 5379.7 | 496.0 | auc=0.983668, logloss=0.149188 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 13126.2 | 13126.2..13126.2 | 1 | 0.681 | - | 13126.2 | - | - (stored whole) | 0.681 (whole/whole) | - | 6475.4 | 496.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host cc560ebdaf91) | lightgbm 4.7.0 | opponent | 4935.5 | 4935.5..4935.5 | 1 | 1.812 | - | 4935.5 | - | - (stored whole) | 1.812 (whole/whole) | - | 4205.8 | 544.0 | auc=0.500000, logloss=0.356515 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on NVIDIA L40S (nv, RunPod) job n0320

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 4128.2 | 4128.2..4128.2 | 1 | - | - | 4128.2 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 15380.3 | 15380.3..15380.3 | 1 | 0.268 | - | 15380.3 | - | - (stored whole) | 0.268 (whole/whole) | - | 1670.4 | 496.0 | auc=0.631766, logloss=0.528065 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host 24a11adce16e) | xgboost 3.2.0 | opponent | 6565.0 | 6565.0..6565.0 | 1 | 0.629 | - | 6565.0 | - | - (stored whole) | 0.629 (whole/whole) | - | 1720.3 | 496.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host 24a11adce16e) | lightgbm 4.7.0 | opponent | 1722.8 | 1722.8..1722.8 | 1 | 2.396 | - | 1722.8 | - | - (stored whole) | 2.396 (whole/whole) | - | 1314.0 | 544.0 | auc=0.500000, logloss=0.554692 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 14667.8 | 14667.8..14667.8 | 1 | - | - | 14667.8 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 15195.6 | 15195.6..15195.6 | 1 | 0.965 | - | 15195.6 | - | - (stored whole) | 0.965 (whole/whole) | - | 5364.3 | 558.0 | accuracy=0.907780, mlogloss=0.258149 | yes | NOT-COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 33556.0 | 33556.0..33556.0 | 1 | 0.437 | - | 33556.0 | - | - (stored whole) | 0.437 (whole/whole) | - | 6693.0 | 558.0 | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host cc560ebdaf91) | lightgbm 4.7.0 | opponent | 170658.0 | 170658.0..170658.0 | 1 | 0.086 | - | 170658.0 | - | - (stored whole) | 0.086 (whole/whole) | - | 4394.5 | 936.0 | accuracy=0.911402, mlogloss=0.243459 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 7044.5 | 7044.5..7044.5 | 1 | - | - | 7044.5 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 10221.5 | 10221.5..10221.5 | 1 | 0.689 | - | 10221.5 | - | - (stored whole) | 0.689 (whole/whole) | - | 1717.7 | 610.0 | accuracy=0.599270, mlogloss=1.012704 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host 24a11adce16e) | xgboost 3.2.0 | opponent | 12711.7 | 12711.7..12711.7 | 1 | 0.554 | - | 12711.7 | - | - (stored whole) | 0.554 (whole/whole) | - | 1903.0 | 610.0 | accuracy=0.601200, mlogloss=1.005204 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | NVIDIA L40S (host 24a11adce16e) | lightgbm 4.7.0 | opponent | 76949.9 | 76949.9..76949.9 | 1 | 0.092 | - | 76949.9 | - | - (stored whole) | 0.092 (whole/whole) | - | 1457.7 | 856.0 | accuracy=0.601896, mlogloss=1.003173 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 35370.0 | 35370.0..35370.0 | 1 | - | - | 35370.0 | - | - (stored whole) | - | - | - | - | auc=0.979444, logloss=0.190928 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 39091.7 | 39091.7..39091.7 | 1 | 0.905 | - | 39091.7 | - | - (stored whole) | 0.905 (whole/whole) | - | 4881.6 | 430.0 | auc=0.979432, logloss=0.190474 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 19986.4 | 19986.4..19986.4 | 1 | - | - | 19986.4 | - | - (stored whole) | - | - | - | - | auc=0.629203, logloss=0.529007 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 20288.2 | 20288.2..20288.2 | 1 | 0.985 | - | 20288.2 | - | - (stored whole) | 0.985 (whole/whole) | - | 1205.7 | 430.0 | auc=0.628945, logloss=0.528997 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0653)`, ran on NVIDIA L40S (nv, RunPod) job n0653

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@9499046fa) | identical | 2768.8 | 2768.8..2768.8 | 1 | - | - | 2768.8 | - | - (stored whole) | - | - | - | - | map=0.854545, ndcg10=0.719953, ndcg5=0.650400 | - | main board, one scored run | - | ok (main@9499046fa nv/n0653 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 4241.3 | 4241.3..4241.3 | 1 | 0.653 | - | 4241.3 | - | - (stored whole) | 0.653 (whole/whole) | - | 6504.8 | 556.0 | map=0.853929, ndcg10=0.719621, ndcg5=0.650025 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 5040.9 | 5040.9..5040.9 | 1 | 0.549 | - | 5040.9 | - | - (stored whole) | 0.549 (whole/whole) | - | 7300.1 | 556.0 | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | xgboost 3.2.0 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 6137.4 | 6137.4..6137.4 | 1 | - | - | 6137.4 | - | - (stored whole) | - | - | - | - | auc=0.980129, logloss=0.186686 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host cc560ebdaf91) | catboost 1.2.10 | opponent | 8181.4 | 8181.4..8181.4 | 1 | 0.750 | - | 8181.4 | - | - (stored whole) | 0.750 (whole/whole) | - | 4878.1 | 428.0 | auc=0.980018, logloss=0.187292 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1007)`, ran on NVIDIA L40S (nv2, RunPod) job v1007

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 3503.2 | 3503.2..3503.2 | 1 | - | - | 3503.2 | - | - (stored whole) | - | - | - | - | auc=0.630436, logloss=0.528554 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1007 2026-10-09; identity vs the amd columns: n/a) |
| catboost-gpu | catboost | gpu | NVIDIA L40S (host 24a11adce16e) | catboost 1.2.10 | opponent | 5982.0 | 5982.0..5982.0 | 1 | 0.586 | - | 5982.0 | - | - (stored whole) | 0.586 (whole/whole) | - | 1127.4 | 428.0 | auc=0.630310, logloss=0.528616 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host 24a11adce16e, NVIDIA L40S box) | catboost 1.2.10 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rf / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 30726.6 | 30726.6..30726.6 | 1 | - | - | 30726.6 | - | - (stored whole) | - | - | - | - | auc=0.945385, logloss=0.182017 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| cuml-rf-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 24023.2 | 24023.2..24023.2 | 1 | 1.279 | - | 24023.2 | - | - (stored whole) | 1.279 (whole/whole) | - | 5222.5 | 444.0 | auc=0.945348, logloss=0.182069 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rf / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1008)`, ran on NVIDIA L40S (nv2, RunPod) job v1008

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 9400.8 | 9400.8..9400.8 | 1 | - | - | 9400.8 | - | - (stored whole) | - | - | - | - | auc=0.617838, logloss=0.525953 | - | main board, one scored run | - | ok (main@444f0b96e nv2/v1008 2026-10-09; identity vs the amd columns: n/a) |
| cuml-rf-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 20651.0 | 20651.0..20651.0 | 1 | 0.455 | - | 20651.0 | - | - (stored whole) | 0.455 (whole/whole) | - | 1601.6 | 444.0 | auc=0.617836, logloss=0.525954 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1234)`, ran on NVIDIA L40S (nv2, RunPod) job v1234

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 84826.3 | 84826.3..84826.3 | 1 | - | - | 84826.3 | - | - (stored whole) | - | - | - | - | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1234 2026-10-10; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 73482.1 | 73482.1..73482.1 | 1 | 1.154 | - | 74125.8 | 73482.1 | 643.73 (upload_ms_untimed) | 1.144 (whole/whole) | - | 2599.3 | 1272.0 | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1234)`, ran on NVIDIA L40S (nv2, RunPod) job v1234

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 13336.4 | 13336.4..13336.4 | 1 | - | - | 13336.4 | - | - (stored whole) | - | - | - | - | n_clusters=36, noise_fraction=0.000174, rows=1000000 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1234 2026-10-10; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 436957.2 | 436957.2..436957.2 | 1 | 0.031 | - | 437005.3 | 436957.2 | 48.06 (upload_ms_untimed) | 0.031 (whole/whole) | - | 841.9 | 474.0 | n_clusters=36, noise_fraction=0.000174, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1023)`, ran on NVIDIA L40S (nv2, RunPod) job v1023

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 1774.0 | 1774.0..1774.0 | 1 | - | - | 1774.0 | - | - (stored whole) | - | - | - | - | n_clusters=47, noise_fraction=0.253810, rows=100000 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1023 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 899.7 | 899.7..899.7 | 1 | 1.972 | - | 992.4 | 899.7 | 92.66 (upload_ms_untimed) | 1.788 (whole/whole) | - | 1866.3 | 526.0 | n_clusters=53, noise_fraction=0.256280, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10 (scikit-learn 11: the same core distance, the 10th neighbour besides the point), min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: min_samples: ours and cuML 10, scikit-learn 11. The SAME k-th neighbour: cuML's runner.h:68-80 (ours transcribes it) runs the k-NN at min_samples + 1 including the point itself; scikit-learn's kneighbors(X, min_samples) counts the point itself (its HDBSCAN Notes say so). tools/bench_board_params.py maps both to one canonical value

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1023)`, ran on NVIDIA L40S (nv2, RunPod) job v1023

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 505.9 | 505.9..505.9 | 1 | - | - | 505.9 | - | - (stored whole) | - | - | - | - | n_clusters=161, noise_fraction=0.140550, rows=100000 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1023 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 369.1 | 369.1..369.1 | 1 | 1.371 | - | 381.7 | 369.1 | 12.57 (upload_ms_untimed) | 1.326 (whole/whole) | - | 940.5 | 448.0 | n_clusters=159, noise_fraction=0.130970, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10 (scikit-learn 11: the same core distance, the 10th neighbour besides the point), min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: min_samples: ours and cuML 10, scikit-learn 11. The SAME k-th neighbour: cuML's runner.h:68-80 (ours transcribes it) runs the k-NN at min_samples + 1 including the point itself; scikit-learn's kneighbors(X, min_samples) counts the point itself (its HDBSCAN Notes say so). tools/bench_board_params.py maps both to one canonical value

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1023)`, ran on NVIDIA L40S (nv2, RunPod) job v1023

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 465.8 | 465.8..465.8 | 1 | - | - | 465.8 | - | - (stored whole) | - | - | - | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1023 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 509.0 | 509.0..509.0 | 1 | 0.915 | - | 1782.2 | 509.0 | 1273.19 (upload_ms_untimed) | 0.261 (whole/whole) | - | 4957.5 | 2154.0 | inertia=6.111e+17, n_iter=21 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 849.0 | 849.0..849.0 | 1 | 0.549 | - | 924.5 | 849.0 | 75.47 (upload_ms_untimed) | 0.504 (whole/whole (kernel not derivable)) | - | 3162.6 | 3468.9 | inertia=5.991e+17, n_iter=55 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1023)`, ran on NVIDIA L40S (nv2, RunPod) job v1023

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 189.0 | 189.0..189.0 | 1 | - | - | 189.0 | - | - (stored whole) | - | - | - | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1023 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 306.6 | 306.6..306.6 | 1 | 0.616 | - | 481.2 | 306.6 | 174.58 (upload_ms_untimed) | 0.393 (whole/whole) | - | 1227.4 | 614.0 | inertia=3.06e+08, n_iter=32 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 609.7 | 609.7..609.7 | 1 | 0.310 | - | 618.9 | 609.7 | 9.25 (upload_ms_untimed) | 0.305 (whole/whole (kernel not derivable)) | - | 1251.1 | 438.0 | inertia=3.06e+08, n_iter=58 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1079)`, ran on NVIDIA L40S (nv2, RunPod) job v1079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 46.8 | 46.8..46.8 | 1 | - | - | 46.8 | - | - (stored whole) | - | - | - | - | recall_at_k=0.976250, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1079 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 78.4 | 78.4..78.4 | 1 | 0.596 | - | 421.9 | 78.4 | 343.41 (upload_ms_untimed) | 0.111 (whole/whole) | - | 1622.3 | 772.0 | recall_at_k=0.976402, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 116.0 | 116.0..116.0 | 1 | 0.403 | - | 132.0 | 116.0 | 16.03 (upload_ms_untimed) | 0.354 (whole/whole (kernel not derivable)) | - | 1223.8 | 3816.8 | recall_at_k=0.981012, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1079)`, ran on NVIDIA L40S (nv2, RunPod) job v1079

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 24.6 | 24.6..24.6 | 1 | - | - | 24.6 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999754, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1079 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 33.3 | 33.3..33.3 | 1 | 0.740 | - | 58.4 | 33.3 | 25.06 (upload_ms_untimed) | 0.422 (whole/whole) | - | 818.1 | 452.0 | recall_at_k=0.999742, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 96.5 | 96.5..96.5 | 1 | 0.255 | - | 98.3 | 96.5 | 1.78 (upload_ms_untimed) | 0.251 (whole/whole (kernel not derivable)) | - | 891.5 | 3174.8 | recall_at_k=0.999773, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1064)`, ran on NVIDIA L40S (nv2, RunPod) job v1064

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 656.6 | 656.6..656.6 | 1 | - | - | 656.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1064 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 78.5 | 78.5..78.5 | 1 | 8.361 | - | 1328.2 | 78.5 | 1249.71 (upload_ms_untimed) | 0.494 (whole/whole) | - | 5075.9 | 2154.0 | finite=True, r2=-11031.855105, rmse=87.647429 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 580.8 | 580.8..580.8 | 1 | 1.131 | - | 733.1 | 580.8 | 152.30 (upload_ms_untimed) | 0.896 (whole/whole (kernel not derivable)) | - | 3013.5 | 8894.4 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 19.8 | 19.8..19.8 | 1 | 33.096 | - | 159.6 | 19.8 | 139.80 (upload_ms_untimed) | 4.113 (whole/whole (kernel not derivable)) | - | 3044.8 | 3455.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1064)`, ran on NVIDIA L40S (nv2, RunPod) job v1064

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 15.0 | 15.0..15.0 | 1 | - | - | 15.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908836, rmse=4.696479 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1064 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 22.1 | 22.1..22.1 | 1 | 0.678 | - | 196.3 | 22.1 | 174.26 (upload_ms_untimed) | 0.076 (whole/whole) | - | 1356.3 | 614.0 | finite=True, r2=0.908836, rmse=4.696488 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 39.0 | 39.0..39.0 | 1 | 0.384 | - | 53.6 | 39.0 | 14.66 (upload_ms_untimed) | 0.279 (whole/whole (kernel not derivable)) | - | 1081.6 | 4649.6 | finite=True, r2=0.908836, rmse=4.696480 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 7.655 | - | 16.2 | 2.0 | 14.27 (upload_ms_untimed) | 0.922 (whole/whole (kernel not derivable)) | - | 1098.7 | 376.4 | finite=True, r2=0.908836, rmse=4.696490 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1226)`, ran on NVIDIA L40S (nv2, RunPod) job v1226

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 102.0 | 102.0..102.0 | 1 | - | - | 102.0 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000 | - | main board, one scored run | - | ok (main@a8e0548a5 nv2/v1226 2026-10-10; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 74.6 | 74.6..74.6 | 1 | 1.367 | - | 1328.3 | 74.6 | 1253.74 (upload_ms_untimed) | 0.077 (whole/whole) | - | 5058.4 | 2190.0 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 17.0 | 17.0..17.0 | 1 | 5.992 | - | 93.5 | 17.0 | 76.49 (upload_ms_untimed) | 1.091 (whole/whole (kernel not derivable)) | - | 2998.2 | 3439.6 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1226)`, ran on NVIDIA L40S (nv2, RunPod) job v1226

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 10.4 | 10.4..10.4 | 1 | - | - | 10.4 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=0.999997 | - | main board, one scored run | - | ok (main@a8e0548a5 nv2/v1226 2026-10-10; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 20.5 | 20.5..20.5 | 1 | 0.510 | - | 191.8 | 20.5 | 171.29 (upload_ms_untimed) | 0.054 (whole/whole) | - | 1308.2 | 642.0 | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 7.259 | - | 10.6 | 1.4 | 9.19 (upload_ms_untimed) | 0.982 (whole/whole (kernel not derivable)) | - | 1057.9 | 344.4 | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical, wave 2

### elasticnet / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1021)`, ran on NVIDIA L40S (nv2, RunPod) job v1021

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 96.9 | 96.9..96.9 | 1 | - | - | 96.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.260922, rmse=0.718134 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1021 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 319.5 | 319.5..319.5 | 1 | 0.303 | - | 1024.9 | 319.5 | 705.36 (upload_ms_untimed) | 0.095 (whole/whole) | - | 2918.9 | 1374.0 | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 5.8 | 5.8..5.8 | 1 | 0.965 | - | 63.8 | 5.8 | 57.95 (upload_ms_untimed) | 0.088 (whole/whole) | - | 968.0 | 498.0 | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1023)`, ran on NVIDIA L40S (nv2, RunPod) job v1023

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 322.2 | 322.2..322.2 | 1 | - | - | 322.2 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999925, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1023 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 301.6 | 301.6..301.6 | 1 | 1.068 | - | 635.6 | 301.6 | 333.95 (upload_ms_untimed) | 0.507 (whole/whole) | - | 1730.7 | 770.0 | recall_at_k=0.999975, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1023)`, ran on NVIDIA L40S (nv2, RunPod) job v1023

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 43.1 | 43.1..43.1 | 1 | - | - | 43.1 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999650, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1023 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host cc560ebdaf91) | cuvs 26.08.01 | opponent | 297.5 | 297.5..297.5 | 1 | 0.145 | - | 323.7 | 297.5 | 26.15 (upload_ms_untimed) | 0.133 (whole/whole) | - | 933.3 | 450.0 | recall_at_k=0.999450, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 935.1 | 935.1..935.1 | 1 | 0.113 | - | 1631.6 | 935.1 | 696.44 (upload_ms_untimed) | 0.065 (whole/whole) | - | 2918.9 | 1374.0 | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 10.2 | 10.2..10.2 | 1 | 0.576 | - | 67.5 | 10.2 | 57.21 (upload_ms_untimed) | 0.087 (whole/whole) | - | 968.5 | 498.0 | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 1854.6 | 1854.6..1854.6 | 1 | 0.969 | - | 2556.2 | 1854.6 | 701.60 (upload_ms_untimed) | 0.703 (whole/whole) | - | 3047.8 | 1374.0 | accuracy=0.924430, logloss=0.181268, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 9.7 | 9.7..9.7 | 1 | 2.095 | - | 71.9 | 9.7 | 62.19 (upload_ms_untimed) | 0.283 (whole/whole) | - | 1084.8 | 498.0 | accuracy=0.763350, logloss=0.538986, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1064)`, ran on NVIDIA L40S (nv2, RunPod) job v1064

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 529.3 | 529.3..529.3 | 1 | - | - | 529.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328674, rmse=0.684427 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1064 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 48.2 | 48.2..48.2 | 1 | 10.973 | - | 754.2 | 48.2 | 705.96 (upload_ms_untimed) | 0.702 (whole/whole) | - | 2969.7 | 1410.0 | finite=True, r2=-0.251259, rmse=0.934403 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1064)`, ran on NVIDIA L40S (nv2, RunPod) job v1064

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 4.9 | 4.9..4.9 | 1 | - | - | 4.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1064 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 15.2 | 15.2..15.2 | 1 | 0.321 | - | 71.7 | 15.2 | 56.48 (upload_ms_untimed) | 0.068 (whole/whole) | - | 1000.4 | 534.0 | finite=True, r2=0.908983, rmse=4.805051 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1226)`, ran on NVIDIA L40S (nv2, RunPod) job v1226

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@a8e0548a5) | identical | 54.0 | 54.0..54.0 | 1 | - | - | 54.0 | - | - (stored whole) | - | - | - | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001314 | - | main board, one scored run | - | ok (main@a8e0548a5 nv2/v1226 2026-10-10; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 35.3 | 35.3..35.3 | 1 | 1.529 | - | 699.3 | 35.3 | 664.00 (upload_ms_untimed) | 0.077 (whole/whole) | - | 2749.0 | 1314.0 | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001472 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 5.9 | 5.9..5.9 | 1 | 0.492 | - | 54.4 | 5.9 | 48.44 (upload_ms_untimed) | 0.054 (whole/whole) | - | 982.5 | 516.0 | explained_variance_ratio_sum=0.999964, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Neural

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1049)`, ran on NVIDIA L40S (nv2, RunPod) job v1049

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 45.3 ms) = 1.133; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 49.2 ms) = 1.043. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 51.3 | 51.3..51.3 | 1 | - | - | 51.3 | - | - (stored whole) | - | - | - | - | max_rel_err_vs_fp64=2.399e-07 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1049 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 50.2 | 50.2..50.2 | 1 | 1.024 | - | 50.2 | - | - (stored whole) | 1.024 (whole/whole (kernel not derivable)) | - | 933.6 | 200.1 | max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 46.8 | 46.8..46.8 | 1 | 1.098 | - | 46.8 | - | - (stored whole) | 1.098 (whole/whole (kernel not derivable)) | - | 938.7 | 200.1 | max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 49.2 | 49.2..49.2 | 1 | 1.043 | - | 49.2 | - | - (stored whole) | 1.043 (whole/whole (kernel not derivable)) | - | 1146.1 | 200.1 | max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 47.6 | 47.6..47.6 | 1 | 1.079 | - | 47.6 | - | - (stored whole) | 1.079 (whole/whole (kernel not derivable)) | - | 1144.8 | 200.1 | max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 45.3 | 45.3..45.3 | 1 | 1.133 | - | 45.3 | - | - (stored whole) | 1.133 (whole/whole (kernel not derivable)) | - | 1174.9 | 240.2 | max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 47.4 | 47.4..47.4 | 1 | 1.084 | - | 47.4 | - | - (stored whole) | 1.084 (whole/whole (kernel not derivable)) | - | 1422.4 | 240.2 | max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1232)`, ran on NVIDIA L40S (nv2, RunPod) job v1232

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 42.5 ms) = 1.144; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 45.8 ms) = 1.059. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 48.6 | 48.6..48.6 | 1 | - | - | 48.6 | - | - (stored whole) | - | - | - | - | mean_nll=9.018733 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1232 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 46.5 | 46.5..46.5 | 1 | 1.043 | - | 46.5 | - | - (stored whole) | 1.043 (whole/whole (kernel not derivable)) | - | 1107.8 | 173.2 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 45.2 | 45.2..45.2 | 1 | 1.075 | - | 45.2 | - | - (stored whole) | 1.075 (whole/whole (kernel not derivable)) | - | 1105.0 | 173.2 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 45.8 | 45.8..45.8 | 1 | 1.059 | - | 45.8 | - | - (stored whole) | 1.059 (whole/whole (kernel not derivable)) | - | 1206.6 | 154.7 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 44.1 | 44.1..44.1 | 1 | 1.102 | - | 44.1 | - | - (stored whole) | 1.102 (whole/whole (kernel not derivable)) | - | 1208.3 | 154.7 | mean_nll=9.018732 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 44.3 | 44.3..44.3 | 1 | 1.097 | - | 44.3 | - | - (stored whole) | 1.097 (whole/whole (kernel not derivable)) | - | 1233.3 | 192.5 | mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 42.5 | 42.5..42.5 | 1 | 1.144 | - | 42.5 | - | - (stored whole) | 1.144 (whole/whole (kernel not derivable)) | - | 1378.4 | 192.5 | mean_nll=9.018669 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1232)`, ran on NVIDIA L40S (nv2, RunPod) job v1232

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 10.8 ms) = 3.290; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 21.6 ms) = 1.643. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 35.4 | 35.4..35.4 | 1 | - | - | 35.4 | - | - (stored whole) | - | - | - | - | loss_first_step=9.018733, loss_last_step=8.418446, steps=2 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1232 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 21.9 | 21.9..21.9 | 1 | 1.618 | - | 21.9 | - | - (stored whole) | 1.618 (whole/whole (kernel not derivable)) | - | 1257.2 | 959.2 | loss_first_step=9.018733, loss_last_step=8.418449, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 17.9 | 17.9..17.9 | 1 | 1.984 | - | 17.9 | - | - (stored whole) | 1.984 (whole/whole (kernel not derivable)) | - | 1252.4 | 959.2 | loss_first_step=9.018732, loss_last_step=8.418548, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 21.6 | 21.6..21.6 | 1 | 1.643 | - | 21.6 | - | - (stored whole) | 1.643 (whole/whole (kernel not derivable)) | - | 1224.5 | 720.7 | loss_first_step=9.018734, loss_last_step=8.418448, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 22.1 | 22.1..22.1 | 1 | 1.602 | - | 22.1 | - | - (stored whole) | 1.602 (whole/whole (kernel not derivable)) | - | 1219.0 | 727.2 | loss_first_step=9.018732, loss_last_step=8.418557, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 24.4 | 24.4..24.4 | 1 | 1.451 | - | 24.4 | - | - (stored whole) | 1.451 (whole/whole (kernel not derivable)) | - | 1388.2 | 796.6 | loss_first_step=9.018402, loss_last_step=8.417328, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 10.8 | 10.8..10.8 | 1 | 3.290 | - | 10.8 | - | - (stored whole) | 3.290 (whole/whole (kernel not derivable)) | - | 1394.1 | 546.6 | loss_first_step=9.018669, loss_last_step=8.417006, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v0999)`, ran on NVIDIA L40S (nv2, RunPod) job v0999

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 193.2 ms) = 0.067; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 150.9 ms) = 0.086. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 13.0 | 13.0..13.0 | 1 | - | - | 13.0 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@444f0b96e nv2/v0999 2026-10-09; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 150.9 | 150.9..150.9 | 1 | 0.086 | - | 150.9 | - | - (stored whole) | 0.086 (whole/whole (kernel not derivable)) | - | 1062.2 | 354.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 152.3 | 152.3..152.3 | 1 | 0.085 | - | 152.3 | - | - (stored whole) | 0.085 (whole/whole (kernel not derivable)) | - | 1041.6 | 354.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 193.2 | 193.2..193.2 | 1 | 0.067 | - | 193.2 | - | - (stored whole) | 0.067 (whole/whole (kernel not derivable)) | - | 1189.9 | 307.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 3.5 | 3.5..3.5 | 1 | 3.712 | - | 3.5 | - | - (stored whole) | 3.712 (whole/whole) | - | 1247.2 | 54.6 | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.945e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.9 | 4.9..4.9 | 1 | 2.661 | - | 4.9 | - | - (stored whole) | 2.661 (whole/whole) | - | 1225.3 | 54.6 | max_abs_diff_vs_ours=3.815e-06, max_rel_diff_vs_ours=1.902e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-eager-bf16, mamba-ssm-fp32, mamba-ssm-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v0999)`, ran on NVIDIA L40S (nv2, RunPod) job v0999

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 21.2 ms) = 0.621; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 25.9 ms) = 0.507. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 13.1 | 13.1..13.1 | 1 | - | - | 13.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@444f0b96e nv2/v0999 2026-10-09; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 25.9 | 25.9..25.9 | 1 | 0.507 | - | 25.9 | - | - (stored whole) | 0.507 (whole/whole (kernel not derivable)) | - | 1191.4 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 25.4 | 25.4..25.4 | 1 | 0.517 | - | 25.4 | - | - (stored whole) | 0.517 (whole/whole (kernel not derivable)) | - | 1186.5 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 28.5 | 28.5..28.5 | 1 | 0.462 | - | 28.5 | - | - (stored whole) | 0.462 (whole/whole (kernel not derivable)) | - | 1256.0 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 28.2 | 28.2..28.2 | 1 | 0.467 | - | 28.2 | - | - (stored whole) | 0.467 (whole/whole (kernel not derivable)) | - | 1208.1 | 3228.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 21.2 | 21.2..21.2 | 1 | 0.621 | - | 21.2 | - | - (stored whole) | 0.621 (whole/whole (kernel not derivable)) | - | 1326.1 | 1692.8 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 25.7 | 25.7..25.7 | 1 | 0.511 | - | 25.7 | - | - (stored whole) | 0.511 (whole/whole (kernel not derivable)) | - | 1382.1 | 1692.8 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 3.9 | 3.9..3.9 | 1 | 3.347 | - | 3.9 | - | - (stored whole) | 3.347 (whole/whole) | - | 1460.8 | 58.6 | max_abs_diff_vs_ours=1.907e-06, max_rel_diff_vs_ours=6.485e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 2.9 | 2.9..2.9 | 1 | 4.512 | - | 2.9 | - | - (stored whole) | 4.512 (whole/whole) | - | 1289.7 | 58.6 | max_abs_diff_vs_ours=0.0005126, max_rel_diff_vs_ours=0.0001743 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16, mamba-ssm-fp32, mamba-ssm-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v0999)`, ran on NVIDIA L40S (nv2, RunPod) job v0999

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 16.3 ms) = 0.251; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 12.4 ms) = 0.331. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@444f0b96e) | identical | 4.1 | 4.1..4.1 | 1 | - | - | 4.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@444f0b96e nv2/v0999 2026-10-09; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 84.0 | 84.0..84.0 | 1 | 0.049 | - | 84.0 | - | - (stored whole) | 0.049 (whole/whole (kernel not derivable)) | - | 1191.2 | 219.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 83.4 | 83.4..83.4 | 1 | 0.049 | - | 83.4 | - | - (stored whole) | 0.049 (whole/whole (kernel not derivable)) | - | 1186.5 | 219.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 12.4 | 12.4..12.4 | 1 | 0.331 | - | 12.4 | - | - (stored whole) | 0.331 (whole/whole (kernel not derivable)) | - | 3464.6 | 83.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 16.5 | 16.5..16.5 | 1 | 0.248 | - | 16.5 | - | - (stored whole) | 0.248 (whole/whole (kernel not derivable)) | - | 3072.8 | 83.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 83.9 | 83.9..83.9 | 1 | 0.049 | - | 83.9 | - | - (stored whole) | 0.049 (whole/whole (kernel not derivable)) | - | 1313.3 | 212.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 16.3 | 16.3..16.3 | 1 | 0.251 | - | 16.3 | - | - (stored whole) | 0.251 (whole/whole (kernel not derivable)) | - | 3813.6 | 90.5 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| mamba-ssm-fp32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.5 | 4.5..4.5 | 1 | 0.919 | - | 4.5 | - | - (stored whole) | 0.919 (whole/whole) | - | 1414.9 | 70.0 | max_abs_diff_vs_ours=0.001235, max_rel_diff_vs_ours=0.0005401 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| mamba-ssm-tf32 | mamba-ssm | gpu | NVIDIA L40S (host f003873fb257) | mamba-ssm 2.3.2.post1+ge9594ce1c732 | opponent | 4.1 | 4.1..4.1 | 1 | 0.999 | - | 4.1 | - | - (stored whole) | 0.999 (whole/whole) | - | 1410.9 | 70.0 | max_abs_diff_vs_ours=0.001376, max_rel_diff_vs_ours=0.0006018 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16, mamba-ssm-fp32, mamba-ssm-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0664)`, ran on NVIDIA L40S (nv, RunPod) job n0664

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.0 ms) = 1.239; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.5 ms) = 1.658. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@432d6e8ff) | identical | 2.4 | 2.4..2.4 | 1 | - | - | 2.4 | - | - (stored whole) | - | - | - | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | main board, one scored run | - | ok (main@432d6e8ff nv/n0664 2026-10-09; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 1.658 | - | 1.5 | - | - (stored whole) | 1.658 (whole/whole (kernel not derivable)) | - | 1015.5 | 16.3 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 1.293 | - | 1.9 | - | - (stored whole) | 1.293 (whole/whole (kernel not derivable)) | - | 1016.6 | 16.3 | loss_first_step=1.160392, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 1.234 | - | 2.0 | - | - (stored whole) | 1.234 (whole/whole (kernel not derivable)) | - | 1104.5 | 16.3 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.406 | - | 1.7 | - | - (stored whole) | 1.406 (whole/whole (kernel not derivable)) | - | 1049.0 | 16.3 | loss_first_step=1.160392, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 1.239 | - | 2.0 | - | - (stored whole) | 1.239 (whole/whole (kernel not derivable)) | - | 1229.1 | 16.3 | loss_first_step=1.160498, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.4 | 2.4..2.4 | 1 | 1.030 | - | 2.4 | - | - (stored whole) | 1.030 (whole/whole (kernel not derivable)) | - | 1295.1 | 16.3 | loss_first_step=1.160498, loss_last_step=1.123462, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1049)`, ran on NVIDIA L40S (nv2, RunPod) job v1049

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 5.5 ms) = 0.910; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 12.1 ms) = 0.413. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 5.0 | 5.0..5.0 | 1 | - | - | 5.0 | - | - (stored whole) | - | - | - | - | mean_nll=5.635910 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1049 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 47.0 | 47.0..47.0 | 1 | 0.106 | - | 47.0 | - | - (stored whole) | 0.106 (whole/whole (kernel not derivable)) | - | 1266.2 | 137.4 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 46.6 | 46.6..46.6 | 1 | 0.107 | - | 46.6 | - | - (stored whole) | 0.107 (whole/whole (kernel not derivable)) | - | 1245.0 | 137.4 | mean_nll=5.635948 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 12.1 | 12.1..12.1 | 1 | 0.413 | - | 12.1 | - | - (stored whole) | 0.413 (whole/whole (kernel not derivable)) | - | 1870.8 | 64.2 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.9 | 4.9..4.9 | 1 | 1.010 | - | 4.9 | - | - (stored whole) | 1.010 (whole/whole (kernel not derivable)) | - | 1726.6 | 64.2 | mean_nll=5.635950 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 47.7 | 47.7..47.7 | 1 | 0.105 | - | 47.7 | - | - (stored whole) | 0.105 (whole/whole (kernel not derivable)) | - | 1371.7 | 142.8 | mean_nll=5.635952 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 5.5 | 5.5..5.5 | 1 | 0.910 | - | 5.5 | - | - (stored whole) | 0.910 (whole/whole (kernel not derivable)) | - | 2114.4 | 74.1 | mean_nll=5.635985 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1049)`, ran on NVIDIA L40S (nv2, RunPod) job v1049

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 165.9 ms) = 0.238; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 46.3 ms) = 0.854. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 39.6 | 39.6..39.6 | 1 | - | - | 39.6 | - | - (stored whole) | - | - | - | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1049 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 165.2 | 165.2..165.2 | 1 | 0.240 | - | 165.2 | - | - (stored whole) | 0.240 (whole/whole (kernel not derivable)) | - | 1406.8 | 382.4 | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 147.5 | 147.5..147.5 | 1 | 0.268 | - | 147.5 | - | - (stored whole) | 0.268 (whole/whole (kernel not derivable)) | - | 1400.1 | 382.4 | loss_first_step=5.635948, loss_last_step=4.833591, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 46.3 | 46.3..46.3 | 1 | 0.854 | - | 46.3 | - | - (stored whole) | 0.854 (whole/whole (kernel not derivable)) | - | 2931.2 | 311.2 | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 43.2 | 43.2..43.2 | 1 | 0.917 | - | 43.2 | - | - (stored whole) | 0.917 (whole/whole (kernel not derivable)) | - | 2375.7 | 316.8 | loss_first_step=5.635947, loss_last_step=4.833560, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 165.9 | 165.9..165.9 | 1 | 0.238 | - | 165.9 | - | - (stored whole) | 0.238 (whole/whole (kernel not derivable)) | - | 1525.4 | 330.6 | loss_first_step=5.635952, loss_last_step=4.833967, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from opponents-specific-20261006; measured this run) |

memory, ours, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-grid-logs.txt (nv2/v1049)`, ran on NVIDIA L40S (nv2, RunPod) job v1049

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 3.4 ms) = 1.165; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 3.6 ms) = 1.113. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 4.0 | 4.0..4.0 | 1 | - | - | 4.0 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1049 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.6 | 3.6..3.6 | 1 | 1.113 | - | 3.6 | - | - (stored whole) | 1.113 (whole/whole (kernel not derivable)) | - | 1099.1 | 78.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.6 | 3.6..3.6 | 1 | 1.114 | - | 3.6 | - | - (stored whole) | 1.114 (whole/whole (kernel not derivable)) | - | 1094.7 | 78.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.5 | 4.5..4.5 | 1 | 0.882 | - | 4.5 | - | - (stored whole) | 0.882 (whole/whole (kernel not derivable)) | - | 993.5 | 46.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 0.934 | - | 4.3 | - | - (stored whole) | 0.934 (whole/whole (kernel not derivable)) | - | 950.5 | 46.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 1.165 | - | 3.4 | - | - (stored whole) | 1.165 (whole/whole (kernel not derivable)) | - | 1291.8 | 80.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 1.064 | - | 3.8 | - | - (stored whole) | 1.064 (whole/whole (kernel not derivable)) | - | 1230.9 | 41.8 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1065)`, ran on NVIDIA L40S (nv2, RunPod) job v1065

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 15.3 ms) = 0.897. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 13.7 | 13.7..13.7 | 1 | - | - | 13.7 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1065 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 15.3 | 15.3..15.3 | 1 | 0.897 | - | - | 15.3 | - (stored kernel) | 0.897 (MIXED ours whole / arm kernel) | - | 1035.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 169.7 | 169.7..169.7 | 1 | 0.081 | - | - | 169.7 | - (stored kernel) | 0.081 (MIXED ours whole / arm kernel) | - | 1118.9 | 1024.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adagrad / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 14.9 ms) = 0.570. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 14.9 | 14.9..14.9 | 1 | 0.570 | - | - | 14.9 | - (stored kernel) | 0.570 (MIXED ours whole / arm kernel) | - | 968.5 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 115.6 | 115.6..115.6 | 1 | 0.073 | - | - | 115.6 | - (stored kernel) | 0.073 (MIXED ours whole / arm kernel) | - | 1302.1 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 20.2 ms) = 0.555. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 11.2 | 11.2..11.2 | 1 | - | - | 11.2 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 20.2 | 20.2..20.2 | 1 | 0.555 | - | - | 20.2 | - (stored kernel) | 0.555 (MIXED ours whole / arm kernel) | - | 981.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 146.9 | 146.9..146.9 | 1 | 0.076 | - | - | 146.9 | - (stored kernel) | 0.076 (MIXED ours whole / arm kernel) | - | 1445.6 | 896.0 | rel_fro_vs_torch_eager_fp32=6.395e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.5 ms) = 0.658; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 2.4 ms) = 0.418. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.0 | 1.0..1.0 | 1 | - | - | 1.0 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 0.393 | - | - | 2.5 | - (stored kernel) | 0.393 (MIXED ours whole / arm kernel) | - | 695.3 | 224.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.4 | 2.4..2.4 | 1 | 0.418 | - | - | 2.4 | - (stored kernel) | 0.418 (MIXED ours whole / arm kernel) | - | 924.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 0.776 | - | - | 1.3 | - (stored kernel) | 0.776 (MIXED ours whole / arm kernel) | - | 695.8 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.667 | - | - | 1.5 | - (stored kernel) | 0.667 (MIXED ours whole / arm kernel) | - | 875.6 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.5 | 1.5..1.5 | 1 | 0.658 | - | - | 1.5 | - (stored kernel) | 0.658 (MIXED ours whole / arm kernel) | - | 695.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 0.398 | - | - | 2.5 | - (stored kernel) | 0.398 (MIXED ours whole / arm kernel) | - | 870.2 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.7 ms) = 0.996; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.1 ms) = 1.456. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.7 | 1.7..1.7 | 1 | - | - | 1.7 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.864 | - | - | 1.9 | - (stored kernel) | 0.864 (MIXED ours whole / arm kernel) | - | 711.3 | 541.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 1.456 | - | - | 1.1 | - (stored kernel) | 1.456 (MIXED ours whole / arm kernel) | - | 943.1 | 540.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.730 | - | - | 2.3 | - (stored kernel) | 0.730 (MIXED ours whole / arm kernel) | - | 711.5 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.710 | - | - | 2.3 | - (stored kernel) | 0.710 (MIXED ours whole / arm kernel) | - | 892.9 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.996 | - | - | 1.7 | - (stored kernel) | 0.996 (MIXED ours whole / arm kernel) | - | 711.3 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.994 | - | - | 1.7 | - (stored kernel) | 0.994 (MIXED ours whole / arm kernel) | - | 887.7 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.8 ms) = 0.871; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1.7 ms) = 0.941. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1092/work-batchnorm1d-synthetic-def/batchnorm1d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.781 | - | - | 2.0 | - (stored kernel) | 0.781 (MIXED ours whole / arm kernel) | - | 776.4 | 320.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.941 | - | - | 1.7 | - (stored kernel) | 0.941 (MIXED ours whole / arm kernel) | - | 982.0 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 0.633 | - | - | 2.5 | - (stored kernel) | 0.633 (MIXED ours whole / arm kernel) | - | 776.6 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.2 | 2.2..2.2 | 1 | 0.722 | - | - | 2.2 | - (stored kernel) | 0.722 (MIXED ours whole / arm kernel) | - | 917.4 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.1 | 2.1..2.1 | 1 | 0.764 | - | - | 2.1 | - (stored kernel) | 0.764 (MIXED ours whole / arm kernel) | - | 776.4 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.871 | - | - | 1.8 | - (stored kernel) | 0.871 (MIXED ours whole / arm kernel) | - | 910.1 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.4 ms) = 0.861; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.3 ms) = 0.878. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 1.2 | 1.2..1.2 | 1 | - | - | 1.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1080/work-batchnorm2d-synthetic-def/batchnorm2d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 0.878 | - | - | 1.3 | - (stored kernel) | 0.878 (MIXED ours whole / arm kernel) | - | 761.8 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 0.822 | - | - | 1.4 | - (stored kernel) | 0.822 (MIXED ours whole / arm kernel) | - | 965.4 | 246.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 0.995 | - | - | 1.2 | - (stored kernel) | 0.995 (MIXED ours whole / arm kernel) | - | 761.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 0.721 | - | - | 1.6 | - (stored kernel) | 0.721 (MIXED ours whole / arm kernel) | - | 903.3 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 0.861 | - | - | 1.4 | - (stored kernel) | 0.861 (MIXED ours whole / arm kernel) | - | 761.7 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.617 | - | - | 1.9 | - (stored kernel) | 0.617 (MIXED ours whole / arm kernel) | - | 894.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 52.3 | 52.3..52.3 | 1 | - | - | 52.3 | - | - (stored whole) | - | - | - | - | accuracy=0.794050, logloss=5.350625 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 64.3 | 64.3..64.3 | 1 | 0.813 | - | 766.1 | 64.3 | 701.76 (upload_ms_untimed) | 0.068 (whole/whole) | - | 3129.9 | 1370.0 | accuracy=0.794050, logloss=5.350631 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1027.1 | 1027.1..1027.1 | 1 | 0.051 | - | 1027.1 | 1027.1 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 3823.4 | - | accuracy=0.794050, logloss=4.278741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 13.5 | 13.5..13.5 | 1 | - | - | 13.5 | - | - (stored whole) | - | - | - | - | accuracy=0.755560, logloss=0.557803 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 12.9 | 12.9..12.9 | 1 | 1.047 | - | 70.7 | 12.9 | 57.84 (upload_ms_untimed) | 0.191 (whole/whole) | - | 1172.6 | 494.0 | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 121.5 | 121.5..121.5 | 1 | 0.111 | - | 121.5 | 121.5 | 0.00 (cpu-arm) | 0.111 (whole/whole) | - | 431.8 | - | accuracy=0.755560, logloss=0.557802 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 2.5 | 2.5..2.5 | 1 | 0.061 | - | 716.0 | 2.5 | 713.58 (upload_ms_untimed) | 0.0002077 (whole/whole) | - | 3009.1 | 1440.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 38.0 | 38.0..38.0 | 1 | 0.004 | - | 38.0 | 38.0 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1499.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 2.4 | 2.4..2.4 | 1 | 0.052 | - | 59.7 | 2.4 | 57.33 (upload_ms_untimed) | 0.002 (whole/whole) | - | 892.9 | 486.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2.2 | 2.2..2.2 | 1 | 0.058 | - | 2.2 | 2.2 | 0.00 (cpu-arm) | 0.058 (whole/whole) | - | 267.4 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 21.0 | 21.0..21.0 | 1 | - | - | 21.0 | - | - (stored whole) | - | - | - | - | accuracy=0.838850, logloss=0.412625 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 76.1 | 76.1..76.1 | 1 | 0.276 | - | 104.2 | 76.1 | 28.07 (upload_ms_untimed) | 0.201 (whole/whole) | - | 1007.4 | 476.0 | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 137.8 | 137.8..137.8 | 1 | 0.152 | - | 137.8 | 137.8 | 0.00 (cpu-arm) | 0.152 (whole/whole) | - | 352.7 | - | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on ours and scikit-learn; cuML has no min_categories option

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 17.1 | 17.1..17.1 | 1 | - | - | 17.1 | - | - (stored whole) | - | - | - | - | accuracy=0.765850, logloss=0.538866 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 13.5 | 13.5..13.5 | 1 | 1.267 | - | 43.1 | 13.5 | 29.63 (upload_ms_untimed) | 0.396 (whole/whole) | - | 1109.7 | 460.0 | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 117.0 | 117.0..117.0 | 1 | 0.146 | - | 117.0 | 117.0 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 315.0 | - | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on ours and scikit-learn; cuML has no min_categories option

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on NVIDIA L40S (nv, RunPod) job n0519

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 296.2 | 296.2..296.2 | 1 | - | - | 296.2 | - | - (stored whole) | - | - | - | - | relative_residual=2.9e-07 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 14.0 | 14.0..14.0 | 1 | 21.150 | - | - | 14.0 | - (stored kernel) | 21.150 (MIXED ours whole / arm kernel) | - | 1214.5 | 768.3 | relative_residual=1.509e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 14.9 | 14.9..14.9 | 1 | 19.872 | - | - | 14.9 | - (stored kernel) | 19.872 (MIXED ours whole / arm kernel) | - | 1285.6 | 1234.0 | relative_residual=1.365e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 4494.5 | 4494.5..4494.5 | 1 | 0.066 | - | 4494.5 | 4494.5 | 0.00 (cpu-arm) | 0.066 (whole/whole) | - | 2677.2 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 483.3 ms) = 0.405; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 458.6 ms) = 0.427. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 195.8 | 195.8..195.8 | 1 | - | - | 195.8 | - | - (stored whole) | - | - | - | - | accuracy=1.000000 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 458.6 | 458.6..458.6 | 1 | 0.427 | - | - | 458.6 | - (stored kernel) | 0.427 (MIXED ours whole / arm kernel) | - | 1317.2 | 287.8 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 569.3 | 569.3..569.3 | 1 | 0.344 | - | - | 569.3 | - (stored kernel) | 0.344 (MIXED ours whole / arm kernel) | - | 1531.3 | 214.3 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 475.9 | 475.9..475.9 | 1 | 0.411 | - | - | 475.9 | - (stored kernel) | 0.411 (MIXED ours whole / arm kernel) | - | 1318.3 | 287.8 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 464.8 | 464.8..464.8 | 1 | 0.421 | - | - | 464.8 | - (stored kernel) | 0.421 (MIXED ours whole / arm kernel) | - | 1358.0 | 214.3 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 483.3 | 483.3..483.3 | 1 | 0.405 | - | - | 483.3 | - (stored kernel) | 0.405 (MIXED ours whole / arm kernel) | - | 1552.4 | 204.7 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 683.3 | 683.3..683.3 | 1 | 0.287 | - | - | 683.3 | - (stored kernel) | 0.287 (MIXED ours whole / arm kernel) | - | 1618.5 | 154.0 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 125.4 | 125.4..125.4 | 1 | - | - | 125.4 | - | - (stored whole) | - | - | - | - | accuracy=0.849360, logloss=3.762524 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 56.5 | 56.5..56.5 | 1 | 2.222 | - | 785.5 | 56.5 | 729.03 (upload_ms_untimed) | 0.160 (whole/whole) | - | 3934.4 | 1372.0 | accuracy=0.849350, logloss=3.763060 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 278.9 | 278.9..278.9 | 1 | 0.450 | - | 278.9 | 278.9 | 0.00 (cpu-arm) | 0.450 (whole/whole) | - | 3912.0 | - | accuracy=0.849350, logloss=3.174763 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 44.2 | 44.2..44.2 | 1 | - | - | 44.2 | - | - (stored whole) | - | - | - | - | accuracy=0.678020, logloss=0.715492 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 12.3 | 12.3..12.3 | 1 | 3.604 | - | 66.8 | 12.3 | 54.56 (upload_ms_untimed) | 0.661 (whole/whole) | - | 1100.8 | 496.0 | accuracy=0.678060, logloss=0.715531 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 70.7 | 70.7..70.7 | 1 | 0.625 | - | 70.7 | 70.7 | 0.00 (cpu-arm) | 0.625 (whole/whole) | - | 434.5 | - | accuracy=0.678030, logloss=0.715493 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on NVIDIA L40S (nv, RunPod) job n0532

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 161.9 | 161.9..161.9 | 1 | - | - | 161.9 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559491 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 61.1 | 61.1..61.1 | 1 | 2.651 | - | 1467.5 | 61.1 | 1406.40 (upload_ms_untimed) | 0.110 (whole/whole) | - | 4710.5 | 1946.0 | accuracy=0.983067, logloss=0.559490 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 410.6 | 410.6..410.6 | 1 | 0.394 | - | 410.6 | 410.6 | 0.00 (cpu-arm) | 0.394 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 0.8 | 0.8..0.8 | 1 | - | - | 0.8 | - | - (stored whole) | - | - | - | - | n_components=81 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host cc560ebdaf91) | cugraph 26.08.00 | opponent | 19.4 | 19.4..19.4 | 1 | 0.042 | - | 19.4 | 19.4 | 0.02 (upload_ms_untimed) | 0.042 (whole/whole) | - | 959.5 | 432.0 | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 13.2 | 13.2..13.2 | 1 | 0.062 | - | 13.2 | 13.2 | 0.00 (cpu-arm) | 0.062 (whole/whole) | - | 117.1 | - | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | n_components=588 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host 24a11adce16e) | cugraph 26.08.00 | opponent | 23.7 | 23.7..23.7 | 1 | 0.068 | - | 23.7 | 23.7 | 0.01 (upload_ms_untimed) | 0.068 (whole/whole) | - | 934.4 | 432.0 | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 18.7 | 18.7..18.7 | 1 | 0.086 | - | 18.7 | 18.7 | 0.00 (cpu-arm) | 0.086 (whole/whole) | - | 99.0 | - | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 3.3 ms) = 3.085; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 3.8 ms) = 2.644. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 10.1 | 10.1..10.1 | 1 | - | - | 10.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1092/work-conv1d-synthetic-def/conv1d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 2.644 | - | - | 3.8 | - (stored kernel) | 2.644 (MIXED ours whole / arm kernel) | - | 989.0 | 640.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 5.1 | 5.1..5.1 | 1 | 1.967 | - | - | 5.1 | - (stored kernel) | 1.967 (MIXED ours whole / arm kernel) | - | 1180.9 | 640.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.9 | 3.9..3.9 | 1 | 2.583 | - | - | 3.9 | - (stored kernel) | 2.583 (MIXED ours whole / arm kernel) | - | 1005.1 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.1 | 4.1..4.1 | 1 | 2.449 | - | - | 4.1 | - (stored kernel) | 2.449 (MIXED ours whole / arm kernel) | - | 1148.7 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.3 | 3.3..3.3 | 1 | 3.085 | - | - | 3.3 | - (stored kernel) | 3.085 (MIXED ours whole / arm kernel) | - | 979.7 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3417.849541, rel_fro_vs_torch_eager_fp32=0.003296 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 3.017 | - | - | 3.4 | - (stored kernel) | 3.017 (MIXED ours whole / arm kernel) | - | 1181.1 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3524.661064, rel_fro_vs_torch_eager_fp32=0.003294 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### conv2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.3 ms) = 4.722; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 2.3 ms) = 2.732. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 6.3 | 6.3..6.3 | 1 | - | - | 6.3 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1080/work-conv2d-synthetic-def/conv2d-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 2.732 | - | - | 2.3 | - (stored kernel) | 2.732 (MIXED ours whole / arm kernel) | - | 910.5 | 250.5 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 2.532 | - | - | 2.5 | - (stored kernel) | 2.532 (MIXED ours whole / arm kernel) | - | 1131.1 | 346.8 | max_rel_diff_vs_torch_eager_fp32=0.506768, rel_fro_vs_torch_eager_fp32=5.007e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 3.555 | - | - | 1.8 | - (stored kernel) | 3.555 (MIXED ours whole / arm kernel) | - | 927.7 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.2 | 2.2..2.2 | 1 | 2.843 | - | - | 2.2 | - (stored kernel) | 2.843 (MIXED ours whole / arm kernel) | - | 1081.7 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 4.722 | - | - | 1.3 | - (stored kernel) | 4.722 (MIXED ours whole / arm kernel) | - | 971.3 | 273.7 | max_rel_diff_vs_torch_eager_fp32=3418.337554, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 3.604 | - | - | 1.7 | - (stored kernel) | 3.604 (MIXED ours whole / arm kernel) | - | 1180.3 | 272.7 | max_rel_diff_vs_torch_eager_fp32=3433.596343, rel_fro_vs_torch_eager_fp32=0.003380 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0610)`, ran on NVIDIA L40S (nv, RunPod) job n0610

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 676.4 | 676.4..676.4 | 1 | - | - | 676.4 | - | - (stored whole) | - | - | - | - | forecast_rmse=13.931153 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0610 2026-10-09; identity vs the amd columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 173.6 | 173.6..173.6 | 1 | 3.896 | - | 173.6 | 173.6 | 0.00 (cpu-arm) | 3.896 (whole/whole) | - | 58.6 | - | forecast_rmse=26.588738 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 164.3 | 164.3..164.3 | 1 | 4.116 | - | 164.3 | 164.3 | 0.00 (cpu-arm) | 4.116 (whole/whole) | - | 468.0 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0610)`, ran on NVIDIA L40S (nv, RunPod) job n0610

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 743.8 | 743.8..743.8 | 1 | - | - | 743.8 | - | - (stored whole) | - | - | - | - | forecast_rmse=96.690449 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0610 2026-10-09; identity vs the amd columns: n/a) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 173.0 | 173.0..173.0 | 1 | 4.300 | - | 173.0 | 173.0 | 0.00 (cpu-arm) | 4.300 (whole/whole) | - | 58.2 | - | forecast_rmse=196.955273 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 133.9 | 133.9..133.9 | 1 | 5.556 | - | 133.9 | 133.9 | 0.00 (cpu-arm) | 5.556 (whole/whole) | - | 468.2 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.0 ms) = 1.130. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | - | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.0 | 1.0..1.0 | 1 | 1.130 | - | - | 1.0 | - (stored kernel) | 1.130 (MIXED ours whole / arm kernel) | - | 697.8 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 0.920 | - | - | 1.2 | - (stored kernel) | 0.920 (MIXED ours whole / arm kernel) | - | 916.4 | 247.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 1.042 | - | - | 1.1 | - (stored kernel) | 1.042 (MIXED ours whole / arm kernel) | - | 697.7 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 0.867 | - | - | 1.3 | - (stored kernel) | 0.867 (MIXED ours whole / arm kernel) | - | 852.0 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on NVIDIA L40S (nv, RunPod) job n0519

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 21119.9 | 21119.9..21119.9 | 1 | - | - | 21119.9 | - | - (stored whole) | - | - | - | - | max_eigenvalue_error=5.95e-05, relative_residual=5.251e-05 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 81.2 | 81.2..81.2 | 1 | 260.080 | - | - | 81.2 | - (stored kernel) | 260.080 (MIXED ours whole / arm kernel) | - | 937.9 | 450.1 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 80.4 | 80.4..80.4 | 1 | 262.523 | - | - | 80.4 | - (stored kernel) | 262.523 (MIXED ours whole / arm kernel) | - | 753.9 | 1114.0 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 14605.9 | 14605.9..14605.9 | 1 | 1.446 | - | 14605.9 | 14605.9 | 0.00 (cpu-arm) | 1.446 (whole/whole) | - | 945.8 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### embedding / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.9 ms) = 1.434. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 2.7 | 2.7..2.7 | 1 | - | - | 2.7 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 1.434 | - | - | 1.9 | - (stored kernel) | 1.434 (MIXED ours whole / arm kernel) | - | 838.7 | 782.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.1 | 2.1..2.1 | 1 | 1.299 | - | - | 2.1 | - (stored kernel) | 1.299 (MIXED ours whole / arm kernel) | - | 1167.7 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1228)`, ran on NVIDIA L40S (nv2, RunPod) job v1228

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 461.4 | 461.4..461.4 | 1 | - | - | 461.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.326805, rmse=0.685379 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1228 2026-10-10; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 112446.5 | 112446.5..112446.5 | 1 | 0.004 | - | 112446.5 | 112446.5 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 8909.5 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1228)`, ran on NVIDIA L40S (nv2, RunPod) job v1228

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 19.7 | 19.7..19.7 | 1 | - | - | 19.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909004, rmse=4.804486 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1228 2026-10-10; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 507.5 | 507.5..507.5 | 1 | 0.039 | - | 507.5 | 507.5 | 0.00 (cpu-arm) | 0.039 (whole/whole) | - | 645.0 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 125.7 | 125.7..125.7 | 1 | 0.733 | - | 828.2 | 125.7 | 702.51 (upload_ms_untimed) | 0.111 (whole/whole) | - | 2931.8 | 1362.0 | accuracy=0.876570, logloss=3.416741 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 595.4 | 595.4..595.4 | 1 | 0.155 | - | 595.4 | 595.4 | 0.00 (cpu-arm) | 0.155 (whole/whole) | - | 2625.7 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0668)`, ran on NVIDIA L40S (nv, RunPod) job n0668

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@6b882cf81) | identical | 20.0 | 20.0..20.0 | 1 | - | - | 20.0 | - | - (stored whole) | - | - | - | - | accuracy=0.719820, logloss=1.132249 | - | main board, one scored run | - | ok (main@6b882cf81 nv/n0668 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 22.3 | 22.3..22.3 | 1 | 0.897 | - | 79.7 | 22.3 | 57.40 (upload_ms_untimed) | 0.251 (whole/whole) | - | 974.2 | 486.0 | accuracy=0.719810, logloss=1.132317 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 137.0 | 137.0..137.0 | 1 | 0.146 | - | 137.0 | 137.0 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 314.3 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 54.3 | 54.3..54.3 | 1 | - | - | 54.3 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.680693 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 31.4 | 31.4..31.4 | 1 | 1.729 | - | 31.4 | - | - (stored whole) | 1.729 (whole/whole) | - | 1813.0 | 438.0 | mean_abs_distortion=0.443920 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 32.3 | 32.3..32.3 | 1 | 1.684 | - | 32.3 | 32.3 | 0.00 (cpu-arm) | 1.684 (whole/whole) | - | 1134.6 | - | mean_abs_distortion=0.177966 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.345752 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 2.3 | 2.3..2.3 | 1 | 1.498 | - | 2.3 | - | - (stored whole) | 1.498 (whole/whole) | - | 894.4 | 438.0 | mean_abs_distortion=0.302259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.6 | 3.6..3.6 | 1 | 0.957 | - | 3.6 | 3.6 | 0.00 (cpu-arm) | 0.957 (whole/whole) | - | 257.1 | - | mean_abs_distortion=0.339791 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 8.5 ms) = 0.353; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 4.5 ms) = 0.675. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-gcn-istella-def/gcn-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 13.7 | 13.7..13.7 | 1 | 0.220 | - | - | 13.7 | - (stored kernel) | 0.220 (MIXED ours whole / arm kernel) | - | 1426.9 | 1934.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.5 | 4.5..4.5 | 1 | 0.675 | - | - | 4.5 | - (stored kernel) | 0.675 (MIXED ours whole / arm kernel) | - | 1348.5 | 399.4 | max_rel_diff_vs_torch_eager_fp32=0.009646, rel_fro_vs_torch_eager_fp32=1.041e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 13.7 | 13.7..13.7 | 1 | 0.220 | - | - | 13.7 | - (stored kernel) | 0.220 (MIXED ours whole / arm kernel) | - | 1424.0 | 1934.6 | max_rel_diff_vs_torch_eager_fp32=281.122146, rel_fro_vs_torch_eager_fp32=0.0002661 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.2 | 8.2..8.2 | 1 | 0.367 | - | - | 8.2 | - (stored kernel) | 0.367 (MIXED ours whole / arm kernel) | - | 1315.1 | 399.4 | max_rel_diff_vs_torch_eager_fp32=281.117866, rel_fro_vs_torch_eager_fp32=0.0002661 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 13.8 | 13.8..13.8 | 1 | 0.218 | - | - | 13.8 | - (stored kernel) | 0.218 (MIXED ours whole / arm kernel) | - | 1548.3 | 2323.8 | max_rel_diff_vs_torch_eager_fp32=2718.059111, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.5 | 8.5..8.5 | 1 | 0.353 | - | - | 8.5 | - (stored kernel) | 0.353 (MIXED ours whole / arm kernel) | - | 1571.5 | 782.4 | max_rel_diff_vs_torch_eager_fp32=2718.055850, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 9.3 ms) = 0.319; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 3.3 ms) = 0.894. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-gcn-taxi-def/gcn-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 12.6 | 12.6..12.6 | 1 | 0.236 | - | - | 12.6 | - (stored kernel) | 0.236 (MIXED ours whole / arm kernel) | - | 1344.8 | 1589.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 3.3 | 3.3..3.3 | 1 | 0.894 | - | - | 3.3 | - (stored kernel) | 0.894 (MIXED ours whole / arm kernel) | - | 1281.3 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002327, rel_fro_vs_torch_eager_fp32=9.803e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 10.9 | 10.9..10.9 | 1 | 0.271 | - | - | 10.9 | - (stored kernel) | 0.271 (MIXED ours whole / arm kernel) | - | 1347.7 | 1589.7 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=7.223e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 3.7 | 3.7..3.7 | 1 | 0.796 | - | - | 3.7 | - (stored kernel) | 0.796 (MIXED ours whole / arm kernel) | - | 1242.7 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002794, rel_fro_vs_torch_eager_fp32=9.801e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 15.9 | 15.9..15.9 | 1 | 0.186 | - | - | 15.9 | - (stored kernel) | 0.186 (MIXED ours whole / arm kernel) | - | 1464.0 | 1874.9 | max_rel_diff_vs_torch_eager_fp32=1509.509282, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 9.3 | 9.3..9.3 | 1 | 0.319 | - | - | 9.3 | - (stored kernel) | 0.319 (MIXED ours whole / arm kernel) | - | 1468.7 | 591.0 | max_rel_diff_vs_torch_eager_fp32=1509.508234, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.6 ms) = 3.251; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.7 ms) = 2.820. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 2.1 | 2.1..2.1 | 1 | - | - | 2.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1092/work-global-avgpool-synthetic-def/global-avgpool-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 0.7 | 0.7..0.7 | 1 | 2.820 | - | - | 0.7 | - (stored kernel) | 2.820 (MIXED ours whole / arm kernel) | - | 692.6 | 12.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 1.544 | - | - | 1.3 | - (stored kernel) | 1.544 (MIXED ours whole / arm kernel) | - | 894.7 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 0.7 | 0.7..0.7 | 1 | 2.869 | - | - | 0.7 | - (stored kernel) | 2.869 (MIXED ours whole / arm kernel) | - | 692.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 0.9 | 0.9..0.9 | 1 | 2.408 | - | - | 0.9 | - (stored kernel) | 2.408 (MIXED ours whole / arm kernel) | - | 842.8 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 3.251 | - | - | 0.6 | - (stored kernel) | 3.251 (MIXED ours whole / arm kernel) | - | 692.4 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 0.8 | 0.8..0.8 | 1 | 2.598 | - | - | 0.8 | - (stored kernel) | 2.598 (MIXED ours whole / arm kernel) | - | 838.2 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 0.6 ms) = 5.009; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 0.6 ms) = 4.985. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 2.9 | 2.9..2.9 | 1 | - | - | 2.9 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 4.985 | - | - | 0.6 | - (stored kernel) | 4.985 (MIXED ours whole / arm kernel) | - | 669.4 | 12.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.2 | 1.2..1.2 | 1 | 2.373 | - | - | 1.2 | - (stored kernel) | 2.373 (MIXED ours whole / arm kernel) | - | 904.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 5.237 | - | - | 0.6 | - (stored kernel) | 5.237 (MIXED ours whole / arm kernel) | - | 669.8 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.1 | 1.1..1.1 | 1 | 2.642 | - | - | 1.1 | - (stored kernel) | 2.642 (MIXED ours whole / arm kernel) | - | 855.8 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 0.6 | 0.6..0.6 | 1 | 5.009 | - | - | 0.6 | - (stored kernel) | 5.009 (MIXED ours whole / arm kernel) | - | 669.4 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.0 | 1.0..1.0 | 1 | 2.882 | - | - | 1.0 | - (stored kernel) | 2.882 (MIXED ours whole / arm kernel) | - | 852.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 3.3 ms) = 2.250; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 4.2 ms) = 1.765. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 7.5 | 7.5..7.5 | 1 | - | - | 7.5 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-graphsage-istella-def/graphsage-istella-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.6 | 8.6..8.6 | 1 | 0.871 | - | - | 8.6 | - (stored kernel) | 0.871 (MIXED ours whole / arm kernel) | - | 1263.5 | 1667.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.2 | 4.2..4.2 | 1 | 1.765 | - | - | 4.2 | - (stored kernel) | 1.765 (MIXED ours whole / arm kernel) | - | 1310.3 | 403.7 | max_rel_diff_vs_torch_eager_fp32=0.134110, rel_fro_vs_torch_eager_fp32=9.7e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.5 | 8.5..8.5 | 1 | 0.882 | - | - | 8.5 | - (stored kernel) | 0.882 (MIXED ours whole / arm kernel) | - | 1259.5 | 1667.4 | max_rel_diff_vs_torch_eager_fp32=430.934131, rel_fro_vs_torch_eager_fp32=0.0002969 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 1.976 | - | - | 3.8 | - (stored kernel) | 1.976 (MIXED ours whole / arm kernel) | - | 1257.6 | 403.8 | max_rel_diff_vs_torch_eager_fp32=430.934131, rel_fro_vs_torch_eager_fp32=0.0002969 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.1 | 8.1..8.1 | 1 | 0.921 | - | - | 8.1 | - (stored kernel) | 0.921 (MIXED ours whole / arm kernel) | - | 1399.9 | 1642.9 | max_rel_diff_vs_torch_eager_fp32=3907.114267, rel_fro_vs_torch_eager_fp32=0.003366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.3 | 3.3..3.3 | 1 | 2.250 | - | - | 3.3 | - (stored kernel) | 2.250 (MIXED ours whole / arm kernel) | - | 1447.6 | 330.6 | max_rel_diff_vs_torch_eager_fp32=3678.210080, rel_fro_vs_torch_eager_fp32=0.003054 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.7 ms) = 1.022; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.6 ms) = 1.074. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.7 | 1.7..1.7 | 1 | - | - | 1.7 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1090/work-graphsage-taxi-def/graphsage-taxi-host-quality/host.log | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 1.074 | - | - | 1.6 | - (stored kernel) | 1.074 (MIXED ours whole / arm kernel) | - | 1181.3 | 289.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.915 | - | - | 1.9 | - (stored kernel) | 0.915 (MIXED ours whole / arm kernel) | - | 1246.5 | 240.0 | max_rel_diff_vs_torch_eager_fp32=0.238419, rel_fro_vs_torch_eager_fp32=6.954e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.765 | - | - | 2.3 | - (stored kernel) | 0.765 (MIXED ours whole / arm kernel) | - | 1182.5 | 289.1 | max_rel_diff_vs_torch_eager_fp32=0.029851, rel_fro_vs_torch_eager_fp32=4.218e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.866 | - | - | 2.0 | - (stored kernel) | 0.866 (MIXED ours whole / arm kernel) | - | 1240.6 | 240.0 | max_rel_diff_vs_torch_eager_fp32=1475.334167, rel_fro_vs_torch_eager_fp32=0.0002145 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.022 | - | - | 1.7 | - (stored kernel) | 1.022 (MIXED ours whole / arm kernel) | - | 1316.0 | 262.6 | max_rel_diff_vs_torch_eager_fp32=5460.333333, rel_fro_vs_torch_eager_fp32=0.003557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.012 | - | - | 1.7 | - (stored kernel) | 1.012 (MIXED ours whole / arm kernel) | - | 1364.9 | 167.2 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1311.6 ms) = 0.341; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1103.7 ms) = 0.405. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.1 | 447.1..447.1 | 1 | - | - | 447.1 | - | - (stored whole) | - | - | - | - | accuracy=0.971625, logloss=0.070054 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1103.7 | 1103.7..1103.7 | 1 | 0.405 | - | - | 1103.7 | - (stored kernel) | 0.405 (MIXED ours whole / arm kernel) | - | 1122.3 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1190.3 | 1190.3..1190.3 | 1 | 0.376 | - | - | 1190.3 | - (stored kernel) | 0.376 (MIXED ours whole / arm kernel) | - | 1174.3 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1063.3 | 1063.3..1063.3 | 1 | 0.420 | - | - | 1063.3 | - (stored kernel) | 0.420 (MIXED ours whole / arm kernel) | - | 1123.7 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1248.7 | 1248.7..1248.7 | 1 | 0.358 | - | - | 1248.7 | - (stored kernel) | 0.358 (MIXED ours whole / arm kernel) | - | 1174.6 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1548.3 | 1548.3..1548.3 | 1 | 0.289 | - | - | 1548.3 | - (stored kernel) | 0.289 (MIXED ours whole / arm kernel) | - | 1487.8 | 340.9 | accuracy=0.971951 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1311.6 | 1311.6..1311.6 | 1 | 0.341 | - | - | 1311.6 | - (stored kernel) | 0.341 (MIXED ours whole / arm kernel) | - | 1539.4 | 340.9 | accuracy=0.971951 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 789.0 ms) = 0.567; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 591.3 ms) = 0.757. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.7 | 447.7..447.7 | 1 | - | - | 447.7 | - | - (stored whole) | - | - | - | - | accuracy=0.865723, logloss=0.303537 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 591.3 | 591.3..591.3 | 1 | 0.757 | - | - | 591.3 | - (stored kernel) | 0.757 (MIXED ours whole / arm kernel) | - | 1122.2 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 662.7 | 662.7..662.7 | 1 | 0.676 | - | - | 662.7 | - (stored kernel) | 0.676 (MIXED ours whole / arm kernel) | - | 1173.5 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 641.7 | 641.7..641.7 | 1 | 0.698 | - | - | 641.7 | - (stored kernel) | 0.698 (MIXED ours whole / arm kernel) | - | 1123.2 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 733.4 | 733.4..733.4 | 1 | 0.611 | - | - | 733.4 | - (stored kernel) | 0.611 (MIXED ours whole / arm kernel) | - | 1174.8 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 838.4 | 838.4..838.4 | 1 | 0.534 | - | - | 838.4 | - (stored kernel) | 0.534 (MIXED ours whole / arm kernel) | - | 1487.9 | 340.9 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 789.0 | 789.0..789.0 | 1 | 0.567 | - | - | 789.0 | - (stored kernel) | 0.567 (MIXED ours whole / arm kernel) | - | 1539.0 | 340.9 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1333.4 ms) = 0.335; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1133.5 ms) = 0.394. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 446.3 | 446.3..446.3 | 1 | - | - | 446.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.982337, rmse=0.153975 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1133.5 | 1133.5..1133.5 | 1 | 0.394 | - | - | 1133.5 | - (stored kernel) | 0.394 (MIXED ours whole / arm kernel) | - | 1136.9 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1143.5 | 1143.5..1143.5 | 1 | 0.390 | - | - | 1143.5 | - (stored kernel) | 0.390 (MIXED ours whole / arm kernel) | - | 1188.0 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1020.0 | 1020.0..1020.0 | 1 | 0.438 | - | - | 1020.0 | - (stored kernel) | 0.438 (MIXED ours whole / arm kernel) | - | 1137.7 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1145.8 | 1145.8..1145.8 | 1 | 0.390 | - | - | 1145.8 | - (stored kernel) | 0.390 (MIXED ours whole / arm kernel) | - | 1190.4 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1528.3 | 1528.3..1528.3 | 1 | 0.292 | - | - | 1528.3 | - (stored kernel) | 0.292 (MIXED ours whole / arm kernel) | - | 1534.4 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1333.4 | 1333.4..1333.4 | 1 | 0.335 | - | - | 1333.4 | - (stored kernel) | 0.335 (MIXED ours whole / arm kernel) | - | 1585.9 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 696.1 ms) = 0.642; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 650.5 ms) = 0.687. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 447.0 | 447.0..447.0 | 1 | - | - | 447.0 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.747875, rmse=0.544554 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 650.5 | 650.5..650.5 | 1 | 0.687 | - | - | 650.5 | - (stored kernel) | 0.687 (MIXED ours whole / arm kernel) | - | 1137.2 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 859.6 | 859.6..859.6 | 1 | 0.520 | - | - | 859.6 | - (stored kernel) | 0.520 (MIXED ours whole / arm kernel) | - | 1188.0 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 545.3 | 545.3..545.3 | 1 | 0.820 | - | - | 545.3 | - (stored kernel) | 0.820 (MIXED ours whole / arm kernel) | - | 1138.4 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 721.3 | 721.3..721.3 | 1 | 0.620 | - | - | 721.3 | - (stored kernel) | 0.620 (MIXED ours whole / arm kernel) | - | 1190.0 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 696.1 | 696.1..696.1 | 1 | 0.642 | - | - | 696.1 | - (stored kernel) | 0.642 (MIXED ours whole / arm kernel) | - | 1534.1 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 762.0 | 762.0..762.0 | 1 | 0.587 | - | - | 762.0 | - (stored kernel) | 0.587 (MIXED ours whole / arm kernel) | - | 1585.0 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 1515.2 | 1515.2..1515.2 | 1 | 0.570 | - | 2206.1 | 1515.2 | 690.93 (upload_ms_untimed) | 0.392 (whole/whole) | - | 3076.8 | 1314.0 | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 17717.5 | 17717.5..17717.5 | 1 | 0.049 | - | 17717.5 | 17717.5 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 2303.9 | - | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### incremental-pca / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 107.6 | 107.6..107.6 | 1 | - | - | 107.6 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=0.999995 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 178.2 | 178.2..178.2 | 1 | 0.604 | - | 231.5 | 178.2 | 53.32 (upload_ms_untimed) | 0.465 (whole/whole) | - | 1176.1 | 518.0 | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 669.5 | 669.5..669.5 | 1 | 0.161 | - | 669.5 | 669.5 | 0.00 (cpu-arm) | 0.161 (whole/whole) | - | 307.8 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1020)`, ran on NVIDIA L40S (nv2, RunPod) job v1020

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 755.7 | 755.7..755.7 | 1 | - | - | 755.7 | - | - (stored whole) | - | - | - | - | recall_at_10=0.802100 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1020 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host cc560ebdaf91) | cuvs 26.08.01 | opponent | 1905.2 | 1905.2..1905.2 | 1 | 0.397 | - | 2248.5 | 1905.2 | 343.31 (upload_ms_untimed) | 0.336 (whole/whole) | - | 1793.1 | 822.0 | recall_at_10=0.791300 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

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
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 471.5 | 471.5..471.5 | 1 | 0.212 | - | 502.2 | 471.5 | 30.72 (upload_ms_untimed) | 0.199 (whole/whole) | - | 995.4 | 466.0 | recall_at_10=0.976875 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 67512.8 | 67512.8..67512.8 | 1 | 0.001 | - | 67512.8 | 67512.8 | 0.00 (cpu-arm) | 0.001 (whole/whole) | - | 135.3 | - | recall_at_10=0.979450 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 2186.4 | 2186.4..2186.4 | 1 | - | - | 2186.4 | - | - (stored whole) | - | - | - | - | recall_at_10=0.809175 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host cc560ebdaf91) | cuvs 26.08.01 | opponent | 1255.0 | 1255.0..1255.0 | 1 | 1.742 | - | 1596.3 | 1255.0 | 341.25 (upload_ms_untimed) | 1.370 (whole/whole) | - | 1815.7 | 822.0 | recall_at_10=0.993625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

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
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 381.6 | 381.6..381.6 | 1 | 0.916 | - | 408.0 | 381.6 | 26.37 (upload_ms_untimed) | 0.857 (whole/whole) | - | 1026.7 | 468.0 | recall_at_10=0.999050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 70811.7 | 70811.7..70811.7 | 1 | 0.005 | - | 70811.7 | 70811.7 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 153.2 | - | recall_at_10=0.999375 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-sq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0318)`, ran on NVIDIA L40S (nv, RunPod) job n0318

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 155.5 | 155.5..155.5 | 1 | - | - | 155.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.934975 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0318 2026-10-08; identity vs the amd columns: n/a) |
| cuvs-gpu | cuvs | gpu | NVIDIA L40S (host 24a11adce16e) | cuvs 26.08.01 | opponent | 138.0 | 138.0..138.0 | 1 | 1.126 | - | 163.5 | 138.0 | 25.50 (upload_ms_untimed) | 0.951 (whole/whole) | - | 960.6 | 464.0 | recall_at_10=0.772500 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | faiss 1.15.1 | opponent | 4836.7 | 4836.7..4836.7 | 1 | 0.032 | - | 4836.7 | 4836.7 | 0.00 (cpu-arm) | 0.032 (whole/whole) | - | 124.2 | - | recall_at_10=0.857050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 235.2 | 235.2..235.2 | 1 | - | - | 235.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 739.0 | 739.0..739.0 | 1 | 0.318 | - | 1447.6 | 739.0 | 708.65 (upload_ms_untimed) | 0.162 (whole/whole) | - | 3068.2 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 5641.8 | 5641.8..5641.8 | 1 | 0.042 | - | 5641.8 | 5641.8 | 0.00 (cpu-arm) | 0.042 (whole/whole) | - | 1459.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 10.2 | 10.2..10.2 | 1 | - | - | 10.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 26.6 | 26.6..26.6 | 1 | 0.382 | - | 85.0 | 26.6 | 58.41 (upload_ms_untimed) | 0.120 (whole/whole) | - | 952.0 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 173.1 | 173.1..173.1 | 1 | 0.059 | - | 173.1 | 173.1 | 0.00 (cpu-arm) | 0.059 (whole/whole) | - | 263.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on NVIDIA L40S (nv, RunPod) job n0583

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 5302.1 | 5302.1..5302.1 | 1 | - | - | 5302.1 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 15329.7 | 15329.7..15329.7 | 1 | 0.346 | - | 15334.1 | 15329.7 | 4.41 (upload_ms_untimed) | 0.346 (whole/whole) | - | 2042.9 | 438.0 | rel_error_vs_exact=0.032177 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 25374.0 | 25374.0..25374.0 | 1 | 0.209 | - | 25374.0 | 25374.0 | 0.00 (cpu-arm) | 0.209 (whole/whole) | - | 2183.9 | - | rel_error_vs_exact=8.574e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on NVIDIA L40S (nv, RunPod) job n0583

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 319.5 | 319.5..319.5 | 1 | - | - | 319.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.056e-07 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 595.0 | 595.0..595.0 | 1 | 0.537 | - | 599.4 | 595.0 | 4.34 (upload_ms_untimed) | 0.533 (whole/whole) | - | 1034.4 | 438.0 | rel_error_vs_exact=7.647e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 9341.1 | 9341.1..9341.1 | 1 | 0.034 | - | 9341.1 | 9341.1 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 397.1 | - | rel_error_vs_exact=8.154e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1020)`, ran on NVIDIA L40S (nv2, RunPod) job v1020

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@0a7b206f1) | identical | 82.6 | 82.6..82.6 | 1 | - | - | 82.6 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@0a7b206f1 nv2/v1020 2026-10-09; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 23.2 | 23.2..23.2 | 1 | 3.562 | - | 23.2 | 23.2 | 0.00 (cpu-arm) | 3.562 (whole/whole) | - | 3499.1 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 78.7 | 78.7..78.7 | 1 | - | - | 78.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x16 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 18.9 | 18.9..18.9 | 1 | 4.156 | - | 60.1 | 18.9 | 41.15 (upload_ms_untimed) | 1.310 (whole/whole) | - | 1423.1 | 558.0 | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 92.9 | 92.9..92.9 | 1 | 0.847 | - | 92.9 | 92.9 | 0.00 (cpu-arm) | 0.847 (whole/whole) | - | 672.8 | - | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 1302.2 | 1302.2..1302.2 | 1 | - | - | 1302.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x259 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 56.3 | 56.3..56.3 | 1 | 23.149 | - | 99.2 | 56.3 | 42.99 (upload_ms_untimed) | 13.122 (whole/whole) | - | 3476.3 | 1560.0 | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 182.8 | 182.8..182.8 | 1 | 7.123 | - | 182.8 | 182.8 | 0.00 (cpu-arm) | 7.123 (whole/whole) | - | 6590.6 | - | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 8.4 | 8.4..8.4 | 1 | - | - | 8.4 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 7.9 | 7.9..7.9 | 1 | 1.062 | - | 37.3 | 7.9 | 29.43 (upload_ms_untimed) | 0.225 (whole/whole) | - | 1022.1 | 490.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 28.5 | 28.5..28.5 | 1 | 0.296 | - | 28.5 | 28.5 | 0.00 (cpu-arm) | 0.296 (whole/whole) | - | 308.0 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 9.4 | 9.4..9.4 | 1 | - | - | 9.4 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 38.9 | 38.9..38.9 | 1 | 0.242 | - | 68.5 | 38.9 | 29.58 (upload_ms_untimed) | 0.137 (whole/whole) | - | 1025.1 | 472.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 110.3 | 110.3..110.3 | 1 | 0.085 | - | 110.3 | 110.3 | 0.00 (cpu-arm) | 0.085 (whole/whole) | - | 295.4 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 194.3 | 194.3..194.3 | 1 | - | - | 194.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.309043, rmse=0.694362 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 43.4 | 43.4..43.4 | 1 | 4.472 | - | 745.1 | 43.4 | 701.69 (upload_ms_untimed) | 0.261 (whole/whole) | - | 3008.6 | 1374.0 | finite=True, r2=0.328088, rmse=0.684726 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 922.6 | 922.6..922.6 | 1 | 0.211 | - | 922.6 | 922.6 | 0.00 (cpu-arm) | 0.211 (whole/whole) | - | 1975.3 | - | finite=True, r2=-4.245e+13, rmse=5.442e+06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 60.8 | 60.8..60.8 | 1 | - | - | 60.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 4.5 | 4.5..4.5 | 1 | 13.431 | - | 63.0 | 4.5 | 58.47 (upload_ms_untimed) | 0.965 (whole/whole) | - | 1046.3 | 498.0 | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 50.8 | 50.8..50.8 | 1 | 1.198 | - | 50.8 | 50.8 | 0.00 (cpu-arm) | 1.198 (whole/whole) | - | 292.2 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1228)`, ran on NVIDIA L40S (nv2, RunPod) job v1228

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 429.1 | 429.1..429.1 | 1 | - | - | 429.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.325506, rmse=0.686040 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1228 2026-10-10; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 97412.0 | 97412.0..97412.0 | 1 | 0.004 | - | 97412.0 | 97412.0 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 8911.4 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1228)`, ran on NVIDIA L40S (nv2, RunPod) job v1228

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 18.2 | 18.2..18.2 | 1 | - | - | 18.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909038, rmse=4.803593 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1228 2026-10-10; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 313.1 | 313.1..313.1 | 1 | 0.058 | - | 313.1 | 313.1 | 0.00 (cpu-arm) | 0.058 (whole/whole) | - | 610.4 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 1.9 ms) = 0.592; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 3.4 ms) = 0.322. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1080/work-layernorm-synthetic-def/layernorm-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 0.322 | - | - | 3.4 | - (stored kernel) | 0.322 (MIXED ours whole / arm kernel) | - | 734.0 | 320.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 7.1 | 7.1..7.1 | 1 | 0.155 | - | - | 7.1 | - (stored kernel) | 0.155 (MIXED ours whole / arm kernel) | - | 984.7 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.566 | - | - | 2.0 | - (stored kernel) | 0.566 (MIXED ours whole / arm kernel) | - | 734.2 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.558 | - | - | 2.0 | - (stored kernel) | 0.558 (MIXED ours whole / arm kernel) | - | 935.4 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.9 | 1.9..1.9 | 1 | 0.592 | - | - | 1.9 | - (stored kernel) | 0.592 (MIXED ours whole / arm kernel) | - | 734.4 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.0 | 2.0..2.0 | 1 | 0.546 | - | - | 2.0 | - (stored kernel) | 0.546 (MIXED ours whole / arm kernel) | - | 930.1 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

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
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host 24a11adce16e) | cugraph 26.08.00 | opponent | 218.4 | 218.4..218.4 | 1 | 0.819 | - | 218.4 | 218.4 | 0.01 (upload_ms_untimed) | 0.819 (whole/whole) | - | 947.9 | 440.0 | modularity=0.909430, n_communities=41 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 2943.6 | 2943.6..2943.6 | 1 | 0.061 | - | 2943.6 | 2943.6 | 0.00 (cpu-arm) | 0.061 (whole/whole) | - | 239.0 | - | modularity=0.908460, n_communities=40 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 38.2 | 38.2..38.2 | 1 | - | - | 38.2 | - | - (stored whole) | - | - | - | - | modularity=0.941953, n_communities=58 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host cc560ebdaf91) | cugraph 26.08.00 | opponent | 83.3 | 83.3..83.3 | 1 | 0.459 | - | 83.3 | 83.3 | 0.02 (upload_ms_untimed) | 0.458 (whole/whole) | - | 917.3 | 436.0 | modularity=0.941795, n_communities=62 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 1444.9 | 1444.9..1444.9 | 1 | 0.026 | - | 1444.9 | 1444.9 | 0.00 (cpu-arm) | 0.026 (whole/whole) | - | 201.6 | - | modularity=0.940781, n_communities=56 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 743.9 ms) = 0.735; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 516.0 ms) = 1.059. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 546.6 | 546.6..546.6 | 1 | - | - | 546.6 | - | - (stored whole) | - | - | - | - | accuracy=0.967068, logloss=0.079720 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 516.0 | 516.0..516.0 | 1 | 1.059 | - | - | 516.0 | - (stored kernel) | 1.059 (MIXED ours whole / arm kernel) | - | 1122.6 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 931.7 | 931.7..931.7 | 1 | 0.587 | - | - | 931.7 | - (stored kernel) | 0.587 (MIXED ours whole / arm kernel) | - | 1173.6 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 532.1 | 532.1..532.1 | 1 | 1.027 | - | - | 532.1 | - (stored kernel) | 1.027 (MIXED ours whole / arm kernel) | - | 1124.1 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 675.9 | 675.9..675.9 | 1 | 0.809 | - | - | 675.9 | - (stored kernel) | 0.809 (MIXED ours whole / arm kernel) | - | 1175.8 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 743.9 | 743.9..743.9 | 1 | 0.735 | - | - | 743.9 | - (stored kernel) | 0.735 (MIXED ours whole / arm kernel) | - | 1490.9 | 367.2 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 838.2 | 838.2..838.2 | 1 | 0.652 | - | - | 838.2 | - (stored kernel) | 0.652 (MIXED ours whole / arm kernel) | - | 1542.0 | 367.2 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1735.3 ms) = 0.315; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1246.5 ms) = 0.438. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 546.5 | 546.5..546.5 | 1 | - | - | 546.5 | - | - (stored whole) | - | - | - | - | accuracy=0.870443, logloss=0.297146 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1246.5 | 1246.5..1246.5 | 1 | 0.438 | - | - | 1246.5 | - (stored kernel) | 0.438 (MIXED ours whole / arm kernel) | - | 1122.8 | 695.4 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1257.4 | 1257.4..1257.4 | 1 | 0.435 | - | - | 1257.4 | - (stored kernel) | 0.435 (MIXED ours whole / arm kernel) | - | 1173.7 | 695.4 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1230.4 | 1230.4..1230.4 | 1 | 0.444 | - | - | 1230.4 | - (stored kernel) | 0.444 (MIXED ours whole / arm kernel) | - | 1124.0 | 695.4 | accuracy=0.868164 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1277.4 | 1277.4..1277.4 | 1 | 0.428 | - | - | 1277.4 | - (stored kernel) | 0.428 (MIXED ours whole / arm kernel) | - | 1175.5 | 695.4 | accuracy=0.868164 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1805.5 | 1805.5..1805.5 | 1 | 0.303 | - | - | 1805.5 | - (stored kernel) | 0.303 (MIXED ours whole / arm kernel) | - | 1490.6 | 367.2 | accuracy=0.868327 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1735.3 | 1735.3..1735.3 | 1 | 0.315 | - | - | 1735.3 | - (stored kernel) | 0.315 (MIXED ours whole / arm kernel) | - | 1542.3 | 367.2 | accuracy=0.868327 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 857.8 ms) = 0.636; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 564.5 ms) = 0.966. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 545.6 | 545.6..545.6 | 1 | - | - | 545.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.979398, rmse=0.166295 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 826.4 | 826.4..826.4 | 1 | 0.660 | - | - | 826.4 | - (stored kernel) | 0.660 (MIXED ours whole / arm kernel) | - | 1131.8 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 564.5 | 564.5..564.5 | 1 | 0.966 | - | - | 564.5 | - (stored kernel) | 0.966 (MIXED ours whole / arm kernel) | - | 1183.3 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 801.6 | 801.6..801.6 | 1 | 0.681 | - | - | 801.6 | - (stored kernel) | 0.681 (MIXED ours whole / arm kernel) | - | 1133.8 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 797.0 | 797.0..797.0 | 1 | 0.684 | - | - | 797.0 | - (stored kernel) | 0.684 (MIXED ours whole / arm kernel) | - | 1184.9 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 978.7 | 978.7..978.7 | 1 | 0.557 | - | - | 978.7 | - (stored kernel) | 0.557 (MIXED ours whole / arm kernel) | - | 1536.5 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 857.8 | 857.8..857.8 | 1 | 0.636 | - | - | 857.8 | - (stored kernel) | 0.636 (MIXED ours whole / arm kernel) | - | 1587.9 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1539.5 ms) = 0.355; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 1243.7 ms) = 0.439. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 545.8 | 545.8..545.8 | 1 | - | - | 545.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.754306, rmse=0.537563 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1330.9 | 1330.9..1330.9 | 1 | 0.410 | - | - | 1330.9 | - (stored kernel) | 0.410 (MIXED ours whole / arm kernel) | - | 1131.8 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1243.7 | 1243.7..1243.7 | 1 | 0.439 | - | - | 1243.7 | - (stored kernel) | 0.439 (MIXED ours whole / arm kernel) | - | 1183.5 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1253.8 | 1253.8..1253.8 | 1 | 0.435 | - | - | 1253.8 | - (stored kernel) | 0.435 (MIXED ours whole / arm kernel) | - | 1133.7 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1285.2 | 1285.2..1285.2 | 1 | 0.425 | - | - | 1285.2 | - (stored kernel) | 0.425 (MIXED ours whole / arm kernel) | - | 1185.1 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1711.2 | 1711.2..1711.2 | 1 | 0.319 | - | - | 1711.2 | - (stored kernel) | 0.319 (MIXED ours whole / arm kernel) | - | 1536.4 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1539.5 | 1539.5..1539.5 | 1 | 0.355 | - | - | 1539.5 | - (stored kernel) | 0.355 (MIXED ours whole / arm kernel) | - | 1587.7 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on NVIDIA L40S (nv, RunPod) job n0521

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 413.7 | 413.7..413.7 | 1 | - | - | 413.7 | - | - (stored whole) | - | - | - | - | relative_residual=0.849956 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 220.0 | 220.0..220.0 | 1 | 1.880 | - | - | 220.0 | - (stored kernel) | 1.880 (MIXED ours whole / arm kernel) | - | 1783.9 | 3516.2 | relative_residual=nan | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host 24a11adce16e) | cupy 14.2.0 | opponent | 244.2 | 244.2..244.2 | 1 | 1.694 | - | - | 244.2 | - (stored kernel) | 1.694 (MIXED ours whole / arm kernel) | - | 2740.5 | 9158.0 | relative_residual=0.849957 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 7725.5 | 7725.5..7725.5 | 1 | 0.054 | - | 7725.5 | 7725.5 | 0.00 (cpu-arm) | 0.054 (whole/whole) | - | 4348.1 | - | relative_residual=0.876581 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on NVIDIA L40S (nv, RunPod) job n0521

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 12.1 | 12.1..12.1 | 1 | - | - | 12.1 | - | - (stored whole) | - | - | - | - | relative_residual=0.756366 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 7.1 | 7.1..7.1 | 1 | 1.712 | - | - | 7.1 | - (stored kernel) | 1.712 (MIXED ours whole / arm kernel) | - | 913.0 | 1122.8 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 9.9 | 9.9..9.9 | 1 | 1.222 | - | - | 9.9 | - (stored kernel) | 1.222 (MIXED ours whole / arm kernel) | - | 727.9 | 2778.0 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 183.1 | 183.1..183.1 | 1 | 0.066 | - | 183.1 | 183.1 | 0.00 (cpu-arm) | 0.066 (whole/whole) | - | 279.6 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on NVIDIA L40S (nv, RunPod) job n0519

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1180.8 | 1180.8..1180.8 | 1 | - | - | 1180.8 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 36.1 | 36.1..36.1 | 1 | 32.689 | - | - | 36.1 | - (stored kernel) | 32.689 (MIXED ours whole / arm kernel) | - | 824.5 | 528.3 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host 24a11adce16e) | cupy 14.2.0 | opponent | 39.5 | 39.5..39.5 | 1 | 29.881 | - | - | 39.5 | - (stored kernel) | 29.881 (MIXED ours whole / arm kernel) | - | 875.6 | 1108.0 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| scipy-cpu | scipy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scipy 1.17.1 | opponent | 5417.0 | 5417.0..5417.0 | 1 | 0.218 | - | 5417.0 | 5417.0 | 0.00 (cpu-arm) | 0.218 (whole/whole) | - | 600.1 | - | relative_residual=4.275e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, scipy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on NVIDIA L40S (nv, RunPod) job n0519

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 462.5 | 462.5..462.5 | 1 | - | - | 462.5 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs the amd columns: n/a) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 36.1 | 36.1..36.1 | 1 | 12.808 | - | - | 36.1 | - (stored kernel) | 12.808 (MIXED ours whole / arm kernel) | - | 824.1 | 528.3 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 35.8 | 35.8..35.8 | 1 | 12.907 | - | - | 35.8 | - (stored kernel) | 12.907 (MIXED ours whole / arm kernel) | - | 778.4 | 984.0 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 10448.5 | 10448.5..10448.5 | 1 | 0.044 | - | 10448.5 | 10448.5 | 0.00 (cpu-arm) | 0.044 (whole/whole) | - | 1360.8 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 59.6 | 59.6..59.6 | 1 | - | - | 59.6 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 32.4 | 32.4..32.4 | 1 | 1.841 | - | 750.7 | 32.4 | 718.27 (upload_ms_untimed) | 0.079 (whole/whole) | - | 3171.9 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 153.4 | 153.4..153.4 | 1 | 0.389 | - | 153.4 | 153.4 | 0.00 (cpu-arm) | 0.389 (whole/whole) | - | 2214.7 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 4.0 | 4.0..4.0 | 1 | - | - | 4.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 3.4 | 3.4..3.4 | 1 | 1.169 | - | 60.1 | 3.4 | 56.62 (upload_ms_untimed) | 0.067 (whole/whole) | - | 892.7 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 41.8 | 41.8..41.8 | 1 | 0.096 | - | 41.8 | 41.8 | 0.00 (cpu-arm) | 0.096 (whole/whole) | - | 301.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1.7 ms) = 1.286; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.4 ms) = 1.632. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 2.2 | 2.2..2.2 | 1 | - | - | 2.2 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.4 | 1.4..1.4 | 1 | 1.632 | - | - | 1.4 | - (stored kernel) | 1.632 (MIXED ours whole / arm kernel) | - | 703.9 | 288.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.6 | 1.6..1.6 | 1 | 1.417 | - | - | 1.6 | - (stored kernel) | 1.417 (MIXED ours whole / arm kernel) | - | 946.1 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.3 | 1.3..1.3 | 1 | 1.690 | - | - | 1.3 | - (stored kernel) | 1.690 (MIXED ours whole / arm kernel) | - | 703.6 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 2.7 | 2.7..2.7 | 1 | 0.833 | - | - | 2.7 | - (stored kernel) | 0.833 (MIXED ours whole / arm kernel) | - | 894.9 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 3.8 | 3.8..3.8 | 1 | 0.588 | - | - | 3.8 | - (stored kernel) | 0.588 (MIXED ours whole / arm kernel) | - | 703.6 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 1.286 | - | - | 1.7 | - (stored kernel) | 1.286 (MIXED ours whole / arm kernel) | - | 890.2 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1092)`, ran on NVIDIA L40S (nv2, RunPod) job v1092

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.2 ms) = 0.654; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 1.8 ms) = 0.832. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 1.5 | 1.5..1.5 | 1 | - | - | 1.5 | - | - (stored whole) | - | - | - | - | identical_to=torch-eager-fp32 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1092 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.8 | 1.8..1.8 | 1 | 0.832 | - | - | 1.8 | - (stored kernel) | 0.832 (MIXED ours whole / arm kernel) | - | 719.8 | 639.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.3 | 2.3..2.3 | 1 | 0.650 | - | - | 2.3 | - (stored kernel) | 0.650 (MIXED ours whole / arm kernel) | - | 1221.8 | 552.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1.7 | 1.7..1.7 | 1 | 0.845 | - | - | 1.7 | - (stored kernel) | 0.845 (MIXED ours whole / arm kernel) | - | 720.0 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.5 | 2.5..2.5 | 1 | 0.579 | - | - | 2.5 | - (stored kernel) | 0.579 (MIXED ours whole / arm kernel) | - | 913.7 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.2 | 2.2..2.2 | 1 | 0.654 | - | - | 2.2 | - (stored kernel) | 0.654 (MIXED ours whole / arm kernel) | - | 720.0 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.8 | 2.8..2.8 | 1 | 0.526 | - | - | 2.8 | - (stored kernel) | 0.526 (MIXED ours whole / arm kernel) | - | 908.0 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### moe / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 4.3 ms) = 3.266; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 11.9 ms) = 1.189. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 14.2 | 14.2..14.2 | 1 | - | - | 14.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1080/work-moe-synthetic-def/moe-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 12.0 | 12.0..12.0 | 1 | 1.178 | - | - | 12.0 | - (stored kernel) | 1.178 (MIXED ours whole / arm kernel) | - | 1008.6 | 469.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 11.9 | 11.9..11.9 | 1 | 1.189 | - | - | 11.9 | - (stored kernel) | 1.189 (MIXED ours whole / arm kernel) | - | 1273.6 | 446.7 | max_rel_diff_vs_torch_eager_fp32=0.026193, rel_fro_vs_torch_eager_fp32=1.548e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 9.8 | 9.8..9.8 | 1 | 1.442 | - | - | 9.8 | - (stored kernel) | 1.442 (MIXED ours whole / arm kernel) | - | 1004.4 | 469.1 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 5.1 | 5.1..5.1 | 1 | 2.786 | - | - | 5.1 | - (stored kernel) | 2.786 (MIXED ours whole / arm kernel) | - | 1220.6 | 478.7 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 3.266 | - | - | 4.3 | - (stored kernel) | 3.266 (MIXED ours whole / arm kernel) | - | 1129.6 | 574.3 | max_rel_diff_vs_torch_eager_fp32=22834.612745, rel_fro_vs_torch_eager_fp32=0.055222 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 4.4 | 4.4..4.4 | 1 | 3.216 | - | - | 4.4 | - (stored kernel) | 3.216 (MIXED ours whole / arm kernel) | - | 1397.4 | 433.6 | max_rel_diff_vs_torch_eager_fp32=22851.987745, rel_fro_vs_torch_eager_fp32=0.055199 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 127.3 | 127.3..127.3 | 1 | - | - | 127.3 | - | - (stored whole) | - | - | - | - | accuracy=0.853620, logloss=3.628565 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 56.7 | 56.7..56.7 | 1 | 2.245 | - | 769.6 | 56.7 | 712.86 (upload_ms_untimed) | 0.165 (whole/whole) | - | 3910.6 | 1372.0 | accuracy=0.853620, logloss=3.628599 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 271.1 | 271.1..271.1 | 1 | 0.470 | - | 271.1 | 271.1 | 0.00 (cpu-arm) | 0.470 (whole/whole) | - | 3913.9 | - | accuracy=0.853620, logloss=3.087499 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 42.0 | 42.0..42.0 | 1 | - | - | 42.0 | - | - (stored whole) | - | - | - | - | accuracy=0.723160, logloss=0.590725 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 72.6 | 72.6..72.6 | 1 | 0.578 | - | 127.1 | 72.6 | 54.55 (upload_ms_untimed) | 0.330 (whole/whole) | - | 1077.0 | 496.0 | accuracy=0.723160, logloss=0.590750 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 69.9 | 69.9..69.9 | 1 | 0.600 | - | 69.9 | 69.9 | 0.00 (cpu-arm) | 0.600 (whole/whole) | - | 436.4 | - | accuracy=0.723160, logloss=0.590725 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on NVIDIA L40S (nv, RunPod) job n0532

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 56.2 | 56.2..56.2 | 1 | - | - | 56.2 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559529 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 50.0 | 50.0..50.0 | 1 | 1.123 | - | 1390.0 | 50.0 | 1339.98 (upload_ms_untimed) | 0.040 (whole/whole) | - | 4711.5 | 1946.0 | accuracy=0.983067, logloss=0.559524 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 454.4 | 454.4..454.4 | 1 | 0.124 | - | 454.4 | 454.4 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nadam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 20.8 ms) = 0.521. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 10.8 | 10.8..10.8 | 1 | - | - | 10.8 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 20.8 | 20.8..20.8 | 1 | 0.521 | - | - | 20.8 | - (stored kernel) | 0.521 (MIXED ours whole / arm kernel) | - | 971.0 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 174.5 | 174.5..174.5 | 1 | 0.062 | - | - | 174.5 | - (stored kernel) | 0.062 (MIXED ours whole / arm kernel) | - | 1443.1 | 896.0 | rel_fro_vs_torch_eager_fp32=1.498e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 2.2 | 2.2..2.2 | 1 | 0.078 | - | 705.7 | 2.2 | 703.55 (upload_ms_untimed) | 0.0002393 (whole/whole) | - | 3051.7 | 1450.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 39.2 | 39.2..39.2 | 1 | 0.004 | - | 39.2 | 39.2 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1459.3 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 0.6 | 0.6..0.6 | 1 | 0.175 | - | 54.9 | 0.6 | 54.32 (upload_ms_untimed) | 0.002 (whole/whole) | - | 935.1 | 496.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2.5 | 2.5..2.5 | 1 | 0.043 | - | 2.5 | 2.5 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 265.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 50.0 | 50.0..50.0 | 1 | - | - | 50.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x119 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 52.0 | 52.0..52.0 | 1 | 0.961 | - | 82.4 | 52.0 | 30.41 (upload_ms_untimed) | 0.607 (whole/whole) | - | 1251.6 | 524.0 | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 93.8 | 93.8..93.8 | 1 | 0.533 | - | 93.8 | 93.8 | 0.00 (cpu-arm) | 0.533 (whole/whole) | - | 531.2 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 43.7 | 43.7..43.7 | 1 | - | - | 43.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x508 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 173.9 | 173.9..173.9 | 1 | 0.252 | - | 202.4 | 173.9 | 28.46 (upload_ms_untimed) | 0.216 (whole/whole) | - | 1557.8 | 656.0 | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 48.4 | 48.4..48.4 | 1 | 0.903 | - | 48.4 | 48.4 | 0.00 (cpu-arm) | 0.903 (whole/whole) | - | 1399.8 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0610)`, ran on NVIDIA L40S (nv, RunPod) job n0610

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 299.5 | 299.5..299.5 | 1 | - | - | 299.5 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.438855 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0610 2026-10-09; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 750.7 | 750.7..750.7 | 1 | 0.399 | - | 750.7 | 750.7 | 0.00 (cpu-arm) | 0.399 (whole/whole) | - | 467.7 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0610)`, ran on NVIDIA L40S (nv, RunPod) job n0610

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 539.6 | 539.6..539.6 | 1 | - | - | 539.6 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.150860 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0610 2026-10-09; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1772.3 | 1772.3..1772.3 | 1 | 0.304 | - | 1772.3 | 1772.3 | 0.00 (cpu-arm) | 0.304 (whole/whole) | - | 467.1 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 49.9 | 49.9..49.9 | 1 | - | - | 49.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 100.3 | 100.3..100.3 | 1 | 0.497 | - | 100.3 | 100.3 | 0.00 (cpu-arm) | 0.497 (whole/whole) | - | 259.8 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 32.3 | 32.3..32.3 | 1 | - | - | 32.3 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 48.5 | 48.5..48.5 | 1 | 0.665 | - | 48.5 | 48.5 | 0.00 (cpu-arm) | 0.665 (whole/whole) | - | 240.9 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host cc560ebdaf91) | cugraph 26.08.00 | opponent | 5.0 | 5.0..5.0 | 1 | 0.284 | - | 5.0 | 5.0 | 0.01 (upload_ms_untimed) | 0.283 (whole/whole) | - | 1009.8 | 440.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 274.0 | 274.0..274.0 | 1 | 0.005 | - | 274.0 | 274.0 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 211.2 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on NVIDIA L40S (nv, RunPod) job n0523

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1.3 | 1.3..1.3 | 1 | - | - | 1.3 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | NVIDIA L40S (host 24a11adce16e) | cugraph 26.08.00 | opponent | 5.0 | 5.0..5.0 | 1 | 0.257 | - | 5.0 | 5.0 | 0.02 (upload_ms_untimed) | 0.256 (whole/whole) | - | 1023.2 | 438.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| networkx-cpu | networkx | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | networkx 3.6.1 | opponent | 189.4 | 189.4..189.4 | 1 | 0.007 | - | 189.4 | 189.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 175.9 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on NVIDIA L40S (nv, RunPod) job n0583

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 7393.5 | 7393.5..7393.5 | 1 | - | - | 7393.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 2320.3 | 2320.3..2320.3 | 1 | 3.186 | - | 2324.4 | 2320.3 | 4.11 (upload_ms_untimed) | 3.181 (whole/whole) | - | 1915.3 | 438.0 | rel_error_vs_exact=1.483e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 90631.6 | 90631.6..90631.6 | 1 | 0.082 | - | 90631.6 | 90631.6 | 0.00 (cpu-arm) | 0.082 (whole/whole) | - | 1725.8 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on NVIDIA L40S (nv, RunPod) job n0583

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@03b648834) | identical | 98.0 | 98.0..98.0 | 1 | - | - | 98.0 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 319.6 | 319.6..319.6 | 1 | 0.306 | - | 323.9 | 319.6 | 4.28 (upload_ms_untimed) | 0.302 (whole/whole) | - | 956.8 | 438.0 | rel_error_vs_exact=1.701e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 133.5 | 133.5..133.5 | 1 | 0.733 | - | 133.5 | 133.5 | 0.00 (cpu-arm) | 0.733 (whole/whole) | - | 476.7 | - | rel_error_vs_exact=1.144e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x152 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 1.0 | 1.0..1.0 | 1 | 0.488 | - | 65.0 | 1.0 | 64.00 (upload_ms_untimed) | 0.008 (whole/whole) | - | 1945.9 | 560.0 | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.2 | 3.2..3.2 | 1 | 0.156 | - | 3.2 | 3.2 | 0.00 (cpu-arm) | 0.156 (whole/whole) | - | 1430.3 | - | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 0.3 | 0.3..0.3 | 1 | - | - | 0.3 | - | - (stored whole) | - | - | - | - | output_shape=100000x77 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 1.5 | 1.5..1.5 | 1 | 0.186 | - | 72.6 | 1.5 | 71.09 (upload_ms_untimed) | 0.004 (whole/whole) | - | 944.4 | 510.0 | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2.2 | 2.2..2.2 | 1 | 0.125 | - | 2.2 | 2.2 | 0.00 (cpu-arm) | 0.125 (whole/whole) | - | 369.8 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0318)`, ran on NVIDIA L40S (nv, RunPod) job n0318

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@de2b2b739) | identical | 120.9 | 120.9..120.9 | 1 | - | - | 120.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0318 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 3161.1 | 3161.1..3161.1 | 1 | 0.038 | - | 3215.7 | 3161.1 | 54.57 (upload_ms_untimed) | 0.038 (whole/whole) | - | 942.6 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 4294.8 | 4294.8..4294.8 | 1 | 0.028 | - | 4294.8 | 4294.8 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 397.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on NVIDIA L40S (nv, RunPod) job n0521

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1177.1 | 1177.1..1177.1 | 1 | - | - | 1177.1 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.485e-07 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 227.1 | 227.1..227.1 | 1 | 5.183 | - | - | 227.1 | - (stored kernel) | 5.183 (MIXED ours whole / arm kernel) | - | 1703.4 | 3008.9 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 230.5 | 230.5..230.5 | 1 | 5.106 | - | - | 230.5 | - (stored kernel) | 5.106 (MIXED ours whole / arm kernel) | - | 2486.4 | 5160.0 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 17211.6 | 17211.6..17211.6 | 1 | 0.068 | - | 17211.6 | 17211.6 | 0.00 (cpu-arm) | 0.068 (whole/whole) | - | 8528.1 | - | relative_gram_difference=2.459e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on NVIDIA L40S (nv, RunPod) job n0521

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 57.2 | 57.2..57.2 | 1 | - | - | 57.2 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.714e-07 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 7.0 | 7.0..7.0 | 1 | 8.190 | - | - | 7.0 | - (stored kernel) | 8.190 (MIXED ours whole / arm kernel) | - | 831.1 | 168.1 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host 24a11adce16e) | cupy 14.2.0 | opponent | 6.8 | 6.8..6.8 | 1 | 8.357 | - | - | 6.8 | - (stored kernel) | 8.357 (MIXED ours whole / arm kernel) | - | 653.7 | 726.0 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 444.9 | 444.9..444.9 | 1 | 0.129 | - | 444.9 | 444.9 | 0.00 (cpu-arm) | 0.129 (whole/whole) | - | 475.9 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1228)`, ran on NVIDIA L40S (nv2, RunPod) job v1228

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 161.5 | 161.5..161.5 | 1 | - | - | 161.5 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.0002359 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1228 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 65.7 | 65.7..65.7 | 1 | 2.460 | - | - | 65.7 | - (stored kernel) | 2.460 (MIXED ours whole / arm kernel) | - | 1637.5 | 1012.2 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1589.1 | 1589.1..1589.1 | 1 | 0.102 | - | 1589.1 | 1589.1 | 0.00 (cpu-arm) | 0.102 (whole/whole) | - | 2079.3 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1228)`, ran on NVIDIA L40S (nv2, RunPod) job v1228

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.38 (source build, main@ca25d9321) | identical | 51.4 | 51.4..51.4 | 1 | - | - | 51.4 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.027197 | - | main board, one scored run | - | ok (main@ca25d9321 nv2/v1228 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 32.0 | 32.0..32.0 | 1 | 1.608 | - | - | 32.0 | - (stored kernel) | 1.608 (MIXED ours whole / arm kernel) | - | 838.9 | 198.1 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1017.5 | 1017.5..1017.5 | 1 | 0.051 | - | 1017.5 | 1017.5 | 0.00 (cpu-arm) | 0.051 (whole/whole) | - | 477.8 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1080)`, ran on NVIDIA L40S (nv2, RunPod) job v1080

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 2.8 ms) = 5.700; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 5.3 ms) = 3.077. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@ca8ea1f8d) | identical | 16.2 | 16.2..16.2 | 1 | - | - | 16.2 | - | - (stored whole) | - | - | - | - | error=Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures missing: run tools/neural_fixtures.py generate (tools/neural_fixtures_box.sh generate on a box, then stage) (no fixture directory: pass --neural-fixtures)); see /root/lq/out/v1080/work-resnet-block-synthetic-def/resnet-block-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@ca8ea1f8d nv2/v1080 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 5.3 | 5.3..5.3 | 1 | 3.076 | - | - | 5.3 | - (stored kernel) | 3.076 (MIXED ours whole / arm kernel) | - | 942.5 | 450.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 5.3 | 5.3..5.3 | 1 | 3.077 | - | - | 5.3 | - (stored kernel) | 3.077 (MIXED ours whole / arm kernel) | - | 1146.2 | 409.2 | max_rel_diff_vs_torch_eager_fp32=0.818182, rel_fro_vs_torch_eager_fp32=5.381e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.4 | 4.4..4.4 | 1 | 3.688 | - | - | 4.4 | - (stored kernel) | 3.688 (MIXED ours whole / arm kernel) | - | 959.7 | 554.3 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 4.3 | 4.3..4.3 | 1 | 3.757 | - | - | 4.3 | - (stored kernel) | 3.757 (MIXED ours whole / arm kernel) | - | 1109.0 | 516.8 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 2.8 | 2.8..2.8 | 1 | 5.700 | - | - | 2.8 | - (stored kernel) | 5.700 (MIXED ours whole / arm kernel) | - | 1002.7 | 424.9 | max_rel_diff_vs_torch_eager_fp32=17838.627100, rel_fro_vs_torch_eager_fp32=0.003621 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 3.4 | 3.4..3.4 | 1 | 4.794 | - | - | 3.4 | - (stored kernel) | 4.794 (MIXED ours whole / arm kernel) | - | 1173.4 | 383.4 | max_rel_diff_vs_torch_eager_fp32=18497.318029, rel_fro_vs_torch_eager_fp32=0.003425 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 23244.9 | 23244.9..23244.9 | 1 | - | - | 23244.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328684, rmse=0.684422 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 188507.2 | 188507.2..188507.2 | 1 | 0.123 | - | 188507.2 | 188507.2 | 0.00 (cpu-arm) | 0.123 (whole/whole) | - | 8535.6 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 162.5 | 162.5..162.5 | 1 | - | - | 162.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 2252.8 | 2252.8..2252.8 | 1 | 0.072 | - | 2252.8 | 2252.8 | 0.00 (cpu-arm) | 0.072 (whole/whole) | - | 378.4 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1056)`, ran on NVIDIA L40S (nv2, RunPod) job v1056

headline: ours IDENTICAL / torch bf16: - (no completed torch bf16 arm); fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 13.3 ms) = 0.646. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 8.6 | 8.6..8.6 | 1 | - | - | 8.6 | - | - (stored whole) | - | - | - | - | relative_error_vs_own_host=0.000000 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1056 2026-10-10; identity vs the amd columns: n/a) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 13.3 | 13.3..13.3 | 1 | 0.646 | - | - | 13.3 | - (stored kernel) | 0.646 (MIXED ours whole / arm kernel) | - | 958.5 | 896.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 118.3 | 118.3..118.3 | 1 | 0.072 | - | - | 118.3 | - (stored kernel) | 0.072 (MIXED ours whole / arm kernel) | - | 1308.8 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 873.9 ms) = 0.279; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 904.6 ms) = 0.270. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 244.0 | 244.0..244.0 | 1 | - | - | 244.0 | - | - (stored whole) | - | - | - | - | accuracy=0.967828, logloss=0.079029 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 904.6 | 904.6..904.6 | 1 | 0.270 | - | - | 904.6 | - (stored kernel) | 0.270 (MIXED ours whole / arm kernel) | - | 1122.2 | 404.2 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1154.2 | 1154.2..1154.2 | 1 | 0.211 | - | - | 1154.2 | - (stored kernel) | 0.211 (MIXED ours whole / arm kernel) | - | 1173.6 | 404.2 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 876.8 | 876.8..876.8 | 1 | 0.278 | - | - | 876.8 | - (stored kernel) | 0.278 (MIXED ours whole / arm kernel) | - | 1123.6 | 404.2 | accuracy=0.953505 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1181.2 | 1181.2..1181.2 | 1 | 0.207 | - | - | 1181.2 | - (stored kernel) | 0.207 (MIXED ours whole / arm kernel) | - | 1174.6 | 404.2 | accuracy=0.953505 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1223.4 | 1223.4..1223.4 | 1 | 0.199 | - | - | 1223.4 | - (stored kernel) | 0.199 (MIXED ours whole / arm kernel) | - | 1459.3 | 220.9 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 873.9 | 873.9..873.9 | 1 | 0.279 | - | - | 873.9 | - (stored kernel) | 0.279 (MIXED ours whole / arm kernel) | - | 1510.4 | 220.9 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 642.8 ms) = 0.379; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 507.7 ms) = 0.480. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 243.9 | 243.9..243.9 | 1 | - | - | 243.9 | - | - (stored whole) | - | - | - | - | accuracy=0.862684, logloss=0.313008 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 507.7 | 507.7..507.7 | 1 | 0.480 | - | - | 507.7 | - (stored kernel) | 0.480 (MIXED ours whole / arm kernel) | - | 1122.3 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 667.6 | 667.6..667.6 | 1 | 0.365 | - | - | 667.6 | - (stored kernel) | 0.365 (MIXED ours whole / arm kernel) | - | 1174.0 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 518.2 | 518.2..518.2 | 1 | 0.471 | - | - | 518.2 | - (stored kernel) | 0.471 (MIXED ours whole / arm kernel) | - | 1123.5 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 597.0 | 597.0..597.0 | 1 | 0.409 | - | - | 597.0 | - (stored kernel) | 0.409 (MIXED ours whole / arm kernel) | - | 1174.7 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 786.6 | 786.6..786.6 | 1 | 0.310 | - | - | 786.6 | - (stored kernel) | 0.310 (MIXED ours whole / arm kernel) | - | 1459.1 | 220.9 | accuracy=0.868001 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 642.8 | 642.8..642.8 | 1 | 0.379 | - | - | 642.8 | - (stored kernel) | 0.379 (MIXED ours whole / arm kernel) | - | 1511.1 | 220.9 | accuracy=0.868001 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-compile-bf16, 1442.1 ms) = 0.168; fp32 twin: ours IDENTICAL / torch fp32 (torch-compile-fp32, 902.4 ms) = 0.269. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 242.4 | 242.4..242.4 | 1 | - | - | 242.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.978602, rmse=0.169476 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1095.6 | 1095.6..1095.6 | 1 | 0.221 | - | - | 1095.6 | - (stored kernel) | 0.221 (MIXED ours whole / arm kernel) | - | 1138.3 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 902.4 | 902.4..902.4 | 1 | 0.269 | - | - | 902.4 | - (stored kernel) | 0.269 (MIXED ours whole / arm kernel) | - | 1188.9 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1136.5 | 1136.5..1136.5 | 1 | 0.213 | - | - | 1136.5 | - (stored kernel) | 0.213 (MIXED ours whole / arm kernel) | - | 1133.1 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1161.7 | 1161.7..1161.7 | 1 | 0.209 | - | - | 1161.7 | - (stored kernel) | 0.209 (MIXED ours whole / arm kernel) | - | 1184.7 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1644.0 | 1644.0..1644.0 | 1 | 0.147 | - | - | 1644.0 | - (stored kernel) | 0.147 (MIXED ours whole / arm kernel) | - | 1505.8 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 1442.1 | 1442.1..1442.1 | 1 | 0.168 | - | - | 1442.1 | - (stored kernel) | 0.168 (MIXED ours whole / arm kernel) | - | 1556.6 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1091)`, ran on NVIDIA L40S (nv2, RunPod) job v1091

headline: ours IDENTICAL / torch bf16 (torch-eager-bf16, 612.7 ms) = 0.396; fp32 twin: ours IDENTICAL / torch fp32 (torch-eager-fp32, 664.7 ms) = 0.365. Note: torch bf16 uses tensor cores; IDENTICAL does not (vendor matrix units are not bit-identical across vendors): the gap is the identity tax.

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 242.4 | 242.4..242.4 | 1 | - | - | 242.4 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.743904, rmse=0.548825 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1091 2026-10-10; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 664.7 | 664.7..664.7 | 1 | 0.365 | - | - | 664.7 | - (stored kernel) | 0.365 (MIXED ours whole / arm kernel) | - | 1138.1 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 753.0 | 753.0..753.0 | 1 | 0.322 | - | - | 753.0 | - (stored kernel) | 0.322 (MIXED ours whole / arm kernel) | - | 1189.1 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 724.2 | 724.2..724.2 | 1 | 0.335 | - | - | 724.2 | - (stored kernel) | 0.335 (MIXED ours whole / arm kernel) | - | 1133.0 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-tf32 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 633.1 | 633.1..633.1 | 1 | 0.383 | - | - | 633.1 | - (stored kernel) | 0.383 (MIXED ours whole / arm kernel) | - | 1184.7 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-eager-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 612.7 | 612.7..612.7 | 1 | 0.396 | - | - | 612.7 | - (stored kernel) | 0.396 (MIXED ours whole / arm kernel) | - | 1505.1 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-bf16 | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 877.8 | 877.8..877.8 | 1 | 0.276 | - | - | 877.8 | - (stored kernel) | 0.276 (MIXED ours whole / arm kernel) | - | 1556.6 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 828.4 | 828.4..828.4 | 1 | 0.280 | - | 1526.7 | 828.4 | 698.31 (upload_ms_untimed) | 0.152 (whole/whole) | - | 3064.3 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 6394.7 | 6394.7..6394.7 | 1 | 0.036 | - | 6394.7 | 6394.7 | 0.00 (cpu-arm) | 0.036 (whole/whole) | - | 1455.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 10.0 | 10.0..10.0 | 1 | - | - | 10.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 34.3 | 34.3..34.3 | 1 | 0.291 | - | 91.2 | 34.3 | 56.92 (upload_ms_untimed) | 0.109 (whole/whole) | - | 947.6 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 184.7 | 184.7..184.7 | 1 | 0.054 | - | 184.7 | 184.7 | 0.00 (cpu-arm) | 0.054 (whole/whole) | - | 265.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0606)`, ran on NVIDIA L40S (nv, RunPod) job n0606

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 1588.5 | 1588.5..1588.5 | 1 | - | - | 1588.5 | - | - (stored whole) | - | - | - | - | accuracy=0.920330 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0606 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 4836.8 | 4836.8..4836.8 | 1 | 0.328 | - | 5546.7 | 4836.8 | 709.90 (upload_ms_untimed) | 0.286 (whole/whole) | - | 3017.4 | 1374.0 | accuracy=0.809350 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 36456.8 | 36456.8..36456.8 | 1 | 0.044 | - | 36456.8 | 36456.8 | 0.00 (cpu-arm) | 0.044 (whole/whole) | - | 1138.6 | - | accuracy=0.910200 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 1543.9 | 1543.9..1543.9 | 1 | 0.671 | - | 1597.4 | 1543.9 | 53.49 (upload_ms_untimed) | 0.649 (whole/whole) | - | 1092.9 | 498.0 | accuracy=0.703120 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 14132.8 | 14132.8..14132.8 | 1 | 0.073 | - | 14132.8 | 14132.8 | 0.00 (cpu-arm) | 0.073 (whole/whole) | - | 263.2 | - | accuracy=0.752520 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

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
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 4857.9 | 4857.9..4857.9 | 1 | 0.321 | - | 5558.3 | 4857.9 | 700.38 (upload_ms_untimed) | 0.281 (whole/whole) | - | 2928.5 | 1374.0 | finite=True, r2=0.327768, rmse=0.684889 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 55283.2 | 55283.2..55283.2 | 1 | 0.028 | - | 55283.2 | 55283.2 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 1133.1 | - | finite=True, r2=-2.197e+24, rmse=1.238e+12 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 1024.3 | 1024.3..1024.3 | 1 | - | - | 1024.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908969, rmse=4.805411 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 1579.7 | 1579.7..1579.7 | 1 | 0.648 | - | 1642.0 | 1579.7 | 62.33 (upload_ms_untimed) | 0.624 (whole/whole) | - | 971.4 | 498.0 | finite=True, r2=0.908979, rmse=4.805168 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 13336.4 | 13336.4..13336.4 | 1 | 0.077 | - | 13336.4 | 13336.4 | 0.00 (cpu-arm) | 0.077 (whole/whole) | - | 259.8 | - | finite=True, r2=0.880681, rmse=5.501638 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### simple-imputer / istella (rows full, shape X 1000000x220; X_true 1000000x220; Xq 100000x220; Xq_true 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 240.2 | 240.2..240.2 | 1 | - | - | 240.2 | - | - (stored whole) | - | - | - | - | masked_rmse=346849.129968 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 250.6 | 250.6..250.6 | 1 | 0.959 | - | 981.6 | 250.6 | 731.04 (upload_ms_untimed) | 0.245 (whole/whole) | - | 4914.9 | 1442.0 | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 36684.7 | 36684.7..36684.7 | 1 | 0.007 | - | 36684.7 | 36684.7 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 8060.1 | - | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### simple-imputer / taxi (rows full, shape X 1000000x11; X_true 1000000x11; Xq 100000x11; Xq_true 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 9.8 | 9.8..9.8 | 1 | - | - | 9.8 | - | - (stored whole) | - | - | - | - | masked_rmse=5.985180 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 22.4 | 22.4..22.4 | 1 | 0.440 | - | 84.8 | 22.4 | 62.39 (upload_ms_untimed) | 0.116 (whole/whole) | - | 1043.3 | 488.0 | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 1480.4 | 1480.4..1480.4 | 1 | 0.007 | - | 1480.4 | 1480.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 597.8 | - | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 54.2 | 54.2..54.2 | 1 | - | - | 54.2 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=1.883381 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 108.4 | 108.4..108.4 | 1 | 0.500 | - | 108.4 | - | - (stored whole) | 0.500 (whole/whole) | - | 1993.6 | 430.0 | mean_abs_distortion=0.876709 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 32.5 | 32.5..32.5 | 1 | 1.671 | - | 32.5 | 32.5 | 0.00 (cpu-arm) | 1.671 (whole/whole) | - | 1134.7 | - | mean_abs_distortion=0.474347 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on NVIDIA L40S (nv, RunPod) job n0314

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 4.7 | 4.7..4.7 | 1 | - | - | 4.7 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.147163 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 5.9 | 5.9..5.9 | 1 | 0.794 | - | 5.9 | - | - (stored whole) | 0.794 (whole/whole) | - | 932.9 | 430.0 | mean_abs_distortion=0.266849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 3.9 | 3.9..3.9 | 1 | 1.186 | - | 3.9 | 3.9 | 0.00 (cpu-arm) | 1.186 (whole/whole) | - | 257.7 | - | mean_abs_distortion=0.381016 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 42.9 | 42.9..42.9 | 1 | - | - | 42.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 421.3 | 421.3..421.3 | 1 | 0.102 | - | 1115.7 | 421.3 | 694.41 (upload_ms_untimed) | 0.038 (whole/whole) | - | 3012.0 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 641.8 | 641.8..641.8 | 1 | 0.067 | - | 641.8 | 641.8 | 0.00 (cpu-arm) | 0.067 (whole/whole) | - | 3265.2 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1090)`, ran on NVIDIA L40S (nv2, RunPod) job v1090

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@1665b5626) | identical | 3.0 | 3.0..3.0 | 1 | - | - | 3.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@1665b5626 nv2/v1090 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 53.5 | 53.5..53.5 | 1 | 0.057 | - | 117.1 | 53.5 | 63.57 (upload_ms_untimed) | 0.026 (whole/whole) | - | 932.4 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 91.4 | 91.4..91.4 | 1 | 0.033 | - | 91.4 | 91.4 | 0.00 (cpu-arm) | 0.033 (whole/whole) | - | 358.3 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on NVIDIA L40S (nv, RunPod) job n0521

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 1516.7 | 1516.7..1516.7 | 1 | - | - | 1516.7 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=662.224271, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host 24a11adce16e) | torch 2.13.0+cu129 | opponent | 266.0 | 266.0..266.0 | 1 | 5.703 | - | - | 266.0 | - (stored kernel) | 5.703 (MIXED ours whole / arm kernel) | - | 1883.3 | 5176.3 | max_rel_singular_value_error=8.068917, relative_reconstruction_error_100k_rows=2.861e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host 24a11adce16e) | cupy 14.2.0 | opponent | 242.8 | 242.8..242.8 | 1 | 6.246 | - | - | 242.8 | - (stored kernel) | 6.246 (MIXED ours whole / arm kernel) | - | 2666.2 | 9146.0 | max_rel_singular_value_error=10270.720886, relative_reconstruction_error_100k_rows=2.385e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 13448.4 | 13448.4..13448.4 | 1 | 0.113 | - | 13448.4 | 13448.4 | 0.00 (cpu-arm) | 0.113 (whole/whole) | - | 8679.1 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on NVIDIA L40S (nv, RunPod) job n0521

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@4df610f3b) | identical | 64.2 | 64.2..64.2 | 1 | - | - | 64.2 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | NVIDIA L40S (host cc560ebdaf91) | torch 2.13.0+cu129 | opponent | 8.6 | 8.6..8.6 | 1 | 7.464 | - | - | 8.6 | - (stored kernel) | 7.464 (MIXED ours whole / arm kernel) | - | 850.4 | 1186.8 | max_rel_singular_value_error=2.595e-06, relative_reconstruction_error_100k_rows=7.244e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | NVIDIA L40S (host cc560ebdaf91) | cupy 14.2.0 | opponent | 8.3 | 8.3..8.3 | 1 | 7.753 | - | - | 8.3 | - (stored kernel) | 7.753 (MIXED ours whole / arm kernel) | - | 658.5 | 2766.0 | max_rel_singular_value_error=3.763e-06, relative_reconstruction_error_100k_rows=6.886e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | numpy 2.4.6 | opponent | 319.0 | 319.0..319.0 | 1 | 0.201 | - | 319.0 | 319.0 | 0.00 (cpu-arm) | 0.201 (whole/whole) | - | 450.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0313)`, ran on NVIDIA L40S (nv, RunPod) job n0313

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@89485bc96) | identical | 347.3 | 347.3..347.3 | 1 | - | - | 347.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0313 2026-10-08; identity vs amd-mi325x: MATCH) |
| gpytorch-gpu | gpytorch | gpu | NVIDIA L40S (host 24a11adce16e) | gpytorch 1.15.2 | opponent | 36.2 | 36.2..36.2 | 1 | 9.581 | - | - | 36.2 | - (stored kernel) | 9.581 (MIXED ours whole / arm kernel) | - | 2157.6 | 1492.0 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | gpytorch 1.15.2 | opponent | 14121.1 | 14121.1..14121.1 | 1 | 0.025 | - | 14121.1 | 14121.1 | 0.00 (cpu-arm) | 0.025 (whole/whole) | - | 3161.9 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 173.6 | 173.6..173.6 | 1 | - | - | 173.6 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 530.1 | 530.1..530.1 | 1 | 0.328 | - | 560.7 | 530.1 | 30.57 (upload_ms_untimed) | 0.310 (whole/whole) | - | 1169.3 | 684.0 | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 542.6 | 542.6..542.6 | 1 | 0.320 | - | 542.6 | 542.6 | 0.00 (cpu-arm) | 0.320 (whole/whole) | - | 340.2 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0609)`, ran on NVIDIA L40S (nv, RunPod) job n0609

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 133.9 | 133.9..133.9 | 1 | - | - | 133.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0609 2026-10-09; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 1170.7 | 1170.7..1170.7 | 1 | 0.114 | - | 1205.0 | 1170.7 | 34.30 (upload_ms_untimed) | 0.111 (whole/whole) | - | 1331.0 | 614.0 | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 227.5 | 227.5..227.5 | 1 | 0.588 | - | 227.5 | 227.5 | 0.00 (cpu-arm) | 0.588 (whole/whole) | - | 296.9 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0610)`, ran on NVIDIA L40S (nv, RunPod) job n0610

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 93.8 | 93.8..93.8 | 1 | - | - | 93.8 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436610 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0610 2026-10-09; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 284.0 | 284.0..284.0 | 1 | 0.330 | - | 284.0 | 284.0 | 0.00 (cpu-arm) | 0.330 (whole/whole) | - | 467.4 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 82.5 | 82.5..82.5 | 1 | 1.137 | - | 82.5 | 82.5 | 0.00 (cpu-arm) | 1.137 (whole/whole) | - | 58.3 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0610)`, ran on NVIDIA L40S (nv, RunPod) job n0610

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@d3c0fd72b) | identical | 295.1 | 295.1..295.1 | 1 | - | - | 295.1 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.020604 | - | main board, one scored run | - | ok (main@d3c0fd72b nv/n0610 2026-10-09; identity vs the amd columns: n/a) |
| statsforecast-cpu | statsforecast | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsforecast 2.1.1 | opponent | 1274.4 | 1274.4..1274.4 | 1 | 0.232 | - | 1274.4 | 1274.4 | 0.00 (cpu-arm) | 0.232 (whole/whole) | - | 467.3 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | statsmodels 0.15.0 | opponent | 102.3 | 102.3..102.3 | 1 | 2.885 | - | 102.3 | 102.3 | 0.00 (cpu-arm) | 2.885 (whole/whole) | - | 58.1 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host 24a11adce16e) | xgboost 3.2.0 | opponent | 30.2 | 30.2..30.2 | 1 | 0.575 | - | 30.2 | - | - (stored whole) | 0.575 (whole/whole) | - | 1535.8 | 530.0 | max_additivity_error=1.956e-06 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 822.3 | 822.3..822.3 | 1 | 0.021 | - | 822.3 | 822.3 | 0.00 (cpu-arm) | 0.021 (whole/whole) | - | 1561.9 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | 803.0 | 803.0..803.0 | 1 | 0.022 | - | 803.0 | 803.0 | 0.00 (cpu-arm) | 0.022 (whole/whole) | - | 1439.3 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | 572.6 | 572.6..572.6 | 1 | 0.030 | - | 572.6 | 572.6 | 0.00 (cpu-arm) | 0.030 (whole/whole) | - | 1526.3 | - | max_additivity_error=4.441e-15 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / taxi (rows full, shape X 100000x11; Xq 10000x11; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0530)`, ran on NVIDIA L40S (nv, RunPod) job n0530

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv, RunPod) | mojolearn 0.8.37 (source build, main@42d1e42c6) | identical | 2.2 | 2.2..2.2 | 1 | - | - | 2.2 | - | - (stored whole) | - | - | - | - | max_additivity_error=3.858e-05 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0530 2026-10-08; identity vs amd-mi325x: MATCH) |
| xgboost-gpu | xgboost | gpu | NVIDIA L40S (host cc560ebdaf91) | xgboost 3.2.0 | opponent | 51.2 | 51.2..51.2 | 1 | 0.042 | - | 51.2 | - | - (stored whole) | 0.042 (whole/whole) | - | 485.3 | 500.0 | max_additivity_error=5.402e-05 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | shap 0.51.0 | opponent | 357.4 | 357.4..357.4 | 1 | 0.006 | - | 357.4 | 357.4 | 0.00 (cpu-arm) | 0.006 (whole/whole) | - | 485.3 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | xgboost 3.2.0 | opponent | 338.8 | 338.8..338.8 | 1 | 0.006 | - | 338.8 | 338.8 | 0.00 (cpu-arm) | 0.006 (whole/whole) | - | 374.0 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | lightgbm 4.7.0 | opponent | 234.4 | 234.4..234.4 | 1 | 0.009 | - | 234.4 | 234.4 | 0.00 (cpu-arm) | 0.009 (whole/whole) | - | 263.0 | - | max_additivity_error=5.684e-13 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1051)`, ran on NVIDIA L40S (nv2, RunPod) job v1051

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 1969.2 | 1969.2..1969.2 | 1 | - | - | 1969.2 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.992124 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1051 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host 24a11adce16e) | cuml 26.08.00 | opponent | 467.7 | 467.7..467.7 | 1 | 4.210 | - | 492.2 | 467.7 | 24.52 (upload_ms_untimed) | 4.001 (whole/whole) | - | 966.6 | 464.0 | trustworthiness_k15=0.990033 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 43202.0 | 43202.0..43202.0 | 1 | 0.046 | - | 43202.0 | 43202.0 | 0.00 (cpu-arm) | 0.046 (whole/whole) | - | 330.8 | - | trustworthiness_k15=0.992182 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv2-results.txt (nv2/v1051)`, ran on NVIDIA L40S (nv2, RunPod) job v1051

| arm | library | device | hardware | version | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | NVIDIA L40S (nv2, RunPod) | mojolearn 0.8.37 (source build, main@9f83ea479) | identical | 1851.6 | 1851.6..1851.6 | 1 | - | - | 1851.6 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.998921 | - | main board, one scored run | - | ok (main@9f83ea479 nv2/v1051 2026-10-10; identity vs the amd columns: n/a) |
| cuml-gpu | cuml | gpu | NVIDIA L40S (host cc560ebdaf91) | cuml 26.08.00 | opponent | 396.9 | 396.9..396.9 | 1 | 4.665 | - | 401.6 | 396.9 | 4.72 (upload_ms_untimed) | 4.610 (whole/whole) | - | 918.1 | 448.0 | trustworthiness_k15=0.998353 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | CPU AMD EPYC 9554 64-Core Processor (host f003873fb257, NVIDIA L40S box) | scikit-learn 1.7.2 | opponent | 38518.3 | 38518.3..38518.3 | 1 | 0.048 | - | 38518.3 | 38518.3 | 0.00 (cpu-arm) | 0.048 (whole/whole) | - | 364.1 | - | trustworthiness_k15=0.998823 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

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

