# mojolearn benchmark board

Generated 2026-10-09T04:04:13Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD nvidia-l40s, version label main@03b648834. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 95 races; oldest cell main@8d8771a8e (2026-10-07T19:42:45Z), newest cell main@03b648834 (2026-10-09T00:34:36Z). Boxes: nv (RunPod L40S).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 125 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: one scored run per cell (lq RACE ALGOS lines, lq CMD bench_board summaries); the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 46, n/a 49).

> Opponents: copied from the stored opponent boards (opponents-default-20261006, opponents-specific-20261006, release-board-resume-r2), never re-run here; `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 2 races.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 46, n/a 49.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 75 failed runs.

| lane | dataset | commit | box/job | reason | ok cell on the board |
|---|---|---|---|---|---|
| avgpool1d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| avgpool2d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| batchnorm1d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| batchnorm2d | synthetic | main@483b0db6d | nv/n0568 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| batchnorm2d | synthetic | main@42d1e42c6 | nv/n0526 | not_ready | none |
| batchnorm2d | synthetic | main@4303e1bfb | nv/n0490 | not_ready | none |
| batchnorm2d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| bernoulli-nb | istella | main@89485bc96 | nv/n0314 | error | none |
| bernoulli-nb | taxi | main@89485bc96 | nv/n0314 | error | none |
| categorical-nb | istella | main@89485bc96 | nv/n0314 | error | none |
| categorical-nb | taxi | main@89485bc96 | nv/n0314 | error | none |
| cnn-clf | synthetic | main@de2b2b739 | nv/n0316 | error | none |
| complement-nb | istella | main@de2b2b739 | nv/n0315 | error | none |
| complement-nb | taxi | main@de2b2b739 | nv/n0315 | error | none |
| conv1d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| conv2d | synthetic | main@42d1e42c6 | nv/n0528 | not_ready | none |
| conv2d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| dropout2d | synthetic | main@483b0db6d | nv/n0568 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| dropout2d | synthetic | main@42d1e42c6 | nv/n0526 | not_ready | none |
| dropout2d | synthetic | main@4303e1bfb | nv/n0490 | not_ready | none |
| dropout2d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| embedding | synthetic | main@483b0db6d | nv/n0568 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| embedding | synthetic | main@42d1e42c6 | nv/n0526 | not_ready | none |
| embedding | synthetic | main@4303e1bfb | nv/n0490 | not_ready | none |
| embedding | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| gaussian-nb | istella | main@89485bc96 | nv/n0314 | error | none |
| gaussian-nb | taxi | main@89485bc96 | nv/n0314 | error | none |
| gcn | istella | main@4303e1bfb | nv/n0491 | not_ready | none |
| gcn | istella | main@89485bc96 | nv/n0314 | not_ready | none |
| gcn | taxi | main@4303e1bfb | nv/n0491 | not_ready | none |
| gcn | taxi | main@89485bc96 | nv/n0314 | not_ready | none |
| global-avgpool | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| global-maxpool | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| graphsage | istella | main@4303e1bfb | nv/n0491 | not_ready | none |
| graphsage | istella | main@89485bc96 | nv/n0314 | not_ready | none |
| graphsage | taxi | main@4303e1bfb | nv/n0491 | not_ready | none |
| graphsage | taxi | main@89485bc96 | nv/n0314 | not_ready | none |
| gru-clf | synthetic | main@de2b2b739 | nv/n0317 | error | none |
| gru-clf | taxi-hourly | main@de2b2b739 | nv/n0317 | error | none |
| gru-reg | synthetic | main@de2b2b739 | nv/n0317 | error | none |
| gru-reg | taxi-hourly | main@de2b2b739 | nv/n0317 | error | none |
| layernorm | synthetic | main@483b0db6d | nv/n0568 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| layernorm | synthetic | main@42d1e42c6 | nv/n0526 | not_ready | none |
| layernorm | synthetic | main@4303e1bfb | nv/n0490 | not_ready | none |
| layernorm | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| lstm-clf | synthetic | main@de2b2b739 | nv/n0317 | error | none |
| lstm-clf | taxi-hourly | main@de2b2b739 | nv/n0317 | error | none |
| lstm-reg | synthetic | main@de2b2b739 | nv/n0317 | error | none |
| lstm-reg | taxi-hourly | main@de2b2b739 | nv/n0317 | error | none |
| maxpool1d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| maxpool2d | synthetic | main@4303e1bfb | nv/n0490 | not_ready | none |
| maxpool2d | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| moe | synthetic | main@42d1e42c6 | nv/n0528 | not_ready | none |
| moe | synthetic | main@4303e1bfb | nv/n0490 | not_ready | none |
| moe | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| multinomial-nb | istella | main@de2b2b739 | nv/n0315 | error | none |
| multinomial-nb | taxi | main@de2b2b739 | nv/n0315 | error | none |
| resnet-block | synthetic | main@42d1e42c6 | nv/n0528 | not_ready | none |
| resnet-block | synthetic | main@4303e1bfb | nv/n0490 | not_ready | none |
| resnet-block | synthetic | main@de2b2b739 | nv/n0316 | not_ready | none |
| rnn-clf | synthetic | main@de2b2b739 | nv/n0317 | error | none |
| rnn-clf | taxi-hourly | main@de2b2b739 | nv/n0317 | error | none |
| rnn-reg | synthetic | main@de2b2b739 | nv/n0317 | error | none |
| rnn-reg | taxi-hourly | main@de2b2b739 | nv/n0317 | error | none |
| sgd-clf | istella | main@89485bc96 | nv/n0314 | error | none |
| sgd-clf | taxi | main@89485bc96 | nv/n0314 | error | none |
| standard-scaler | istella | main@89485bc96 | nv/n0314 | error | none |
| standard-scaler | taxi | main@89485bc96 | nv/n0314 | error | none |
| target-encoder | istella | main@89485bc96 | nv/n0314 | error | none |
| target-encoder | taxi | main@89485bc96 | nv/n0314 | error | none |
| conv2d | synthetic | main@e5f3f2ed8 | nv/n0570 | NO-RECORD | none |
| moe | synthetic | main@e5f3f2ed8 | nv/n0570 | NO-RECORD | none |
| resnet-block | synthetic | main@e5f3f2ed8 | nv/n0570 | NO-RECORD | none |
| gbdt-categorical | istella | main@de2b2b739 | nv/n0320 | NO-RECORD | none |
| gbdt-categorical | taxi | main@de2b2b739 | nv/n0320 | REFUSED(Exception_during_warm-up:_At_max/mojo/max/gpu/host/device_context.mojo:4073:35:_CUDA_call_failed:_CUDA_ERROR_OUT_OF_MEMO) | none |

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
| mojolearn | main@03b648834 (wheel none (unreleased; built from source at each cell's commit), sha256 -) |
| script commit | 03b6488348168c2bda73046a3792487b6d7bf47c |
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

Races: 95 planned, 95 done, 0 failed, 0 unsupported, 0 pending. Cells: 304 (REFUSED 22, ok 282).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.395e-09 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | torch-gpu 1.509e-07; cupy-gpu 1.365e-07; numpy-cpu 3.928e-08 |
| algos | complement-nb | text | accuracy (higher is better) | - | 0.983067 | cuml-gpu 0.983067; sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | 0.559491 | cuml-gpu 0.559490; sklearn-cpu 0.557285 |
| algos | connected-components | istella | n_components | - | 81 | cugraph-gpu 81; networkx-cpu 81 |
| algos | connected-components | taxi | n_components | - | 588 | cugraph-gpu 588; networkx-cpu 588 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 5.95e-05 | torch-gpu 1.044e-06; cupy-gpu 1.044e-06; numpy-cpu 3.49e-08 |
| algos | eigh | synthetic | relative_residual | - | 5.251e-05 | torch-gpu 1.016e-06; cupy-gpu 1.016e-06; numpy-cpu 2.824e-08 |
| algos | gaussian-rp | istella | mean_abs_distortion | - | 0.680693 | cuml-gpu 0.443920; sklearn-cpu 0.177966 |
| algos | gaussian-rp | taxi | mean_abs_distortion | - | 0.345752 | cuml-gpu 0.302259; sklearn-cpu 0.339791 |
| algos | incremental-pca | istella | explained_variance_fraction | - | 1.000000 | cuml-gpu 1.000000; sklearn-cpu 1.000000 |
| algos | incremental-pca | taxi | explained_variance_fraction | - | 0.999995 | cuml-gpu 0.999995; sklearn-cpu 0.999995 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | 0.553500 | cuvs-gpu 0.791300; faiss-cpu - |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.966550 | cuvs-gpu 0.976875; faiss-cpu 0.979450 |
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
| algos | louvain | istella | modularity | - | 0.911187 | cugraph-gpu 0.909430; networkx-cpu 0.908460 |
| algos | louvain | istella | n_communities | - | 40 | cugraph-gpu 41; networkx-cpu 40 |
| algos | louvain | taxi | modularity | - | 0.941953 | cugraph-gpu 0.941795; networkx-cpu 0.940781 |
| algos | louvain | taxi | n_communities | - | 58 | cugraph-gpu 62; networkx-cpu 56 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | torch-gpu nan; cupy-gpu 0.849957; numpy-cpu 0.876581 |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | torch-gpu 0.756366; cupy-gpu 0.756366; numpy-cpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 3.386e-07; cupy-gpu 3.386e-07; scipy-cpu 4.275e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 3.386e-07; cupy-gpu 3.386e-07; numpy-cpu 3.259e-08 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | cuml-gpu 0.983067; sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | cuml-gpu 0.559524; sklearn-cpu 0.557319 |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.498e-07 |
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
| algos | ridge-cv | taxi | r2 (higher is better) | - | 0.908981 | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | 4.805109 | sklearn-cpu 4.805055 |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
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
| algos | tree-shap | istella | max_additivity_error | - | 1.175e-06 | xgboost-gpu 1.956e-06; shap-cpu 1.837e-06; xgboost-cpu 1.837e-06; lightgbm-cpu 4.441e-15 |
| algos | tree-shap | taxi | max_additivity_error | - | 3.858e-05 | xgboost-gpu 5.402e-05; shap-cpu 0.0001201; xgboost-cpu 0.0001201; lightgbm-cpu 5.684e-13 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | cuml-gpu 6.111e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | cuml-gpu -; torch-gpu - |
| classical | kmeans | istella | n_iter | - | 33 | cuml-gpu 21; torch-gpu 55 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | cuml-gpu 3.06e+08; torch-gpu 3.06e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | cuml-gpu -; torch-gpu - |
| classical | kmeans | taxi | n_iter | - | 91 | cuml-gpu 32; torch-gpu 58 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | cuml-gpu -11031.855105; torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | cuml-gpu 87.647429; torch-gpu nan; torch-gpu-eigh 0.768590 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908836 | cuml-gpu 0.908836; torch-gpu 0.908836; torch-gpu-eigh 0.908836 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696479 | cuml-gpu 4.696488; torch-gpu 4.696480; torch-gpu-eigh 4.696490 |
| classical2 | gmm | istella | bic (lower is better) | - | -3.851e+07 | - |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | - | 200.794500 | - |
| classical2 | gmm | istella | n_iter | - | 24 | - |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.668e+06 | - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.807640 | - |
| classical2 | gmm | taxi | n_iter | - | 29 | - |
| classical2 | ivf | istella | recall_at_k (higher is better) | - | 1.000000 | cuvs-gpu 0.999975 |
| classical2 | ivf | istella | rows_with_repeated_ids | - | 0 | cuvs-gpu 0 |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | 0.999675 | cuvs-gpu 0.999450 |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | 0 | cuvs-gpu 0 |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328682 | cuml-gpu -0.251259 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684423 | cuml-gpu 0.934403 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | cuml-gpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805050 | cuml-gpu 4.805051 |
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
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-gpu 0.980018; catboost-cpu - |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-gpu 0.187292; catboost-cpu - |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-gpu 0.630310; catboost-cpu - |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-gpu 0.528616; catboost-cpu - |

## Trees

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8797.6 | 8797.6..8797.6 | 1 | - | - | 8797.6 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 9671.9 | 9671.9..9671.9 | 1 | 0.910 | - | 9671.9 | - | - (stored whole) | 0.910 (whole/whole) | - | 5377.9 | 502.0 | auc=0.983152, logloss=0.156748 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 9340.7 | 9340.7..9340.7 | 1 | 0.942 | - | 9340.7 | - | - (stored whole) | 0.942 (whole/whole) | - | 6478.8 | 502.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4061.7 | 4061.7..4061.7 | 1 | - | - | 4061.7 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 6393.8 | 6393.8..6393.8 | 1 | 0.635 | - | 6393.8 | - | - (stored whole) | 0.635 (whole/whole) | - | 1576.3 | 496.0 | auc=0.632335, logloss=0.527912 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 3309.5 | 3309.5..3309.5 | 1 | 1.227 | - | 3309.5 | - | - (stored whole) | 1.227 (whole/whole) | - | 1670.3 | 496.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8941.1 | 8941.1..8941.1 | 1 | - | - | 8941.1 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 23630.6 | 23630.6..23630.6 | 1 | 0.378 | - | 23630.6 | - | - (stored whole) | 0.378 (whole/whole) | - | 5379.7 | 496.0 | auc=0.983668, logloss=0.149188 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 13126.2 | 13126.2..13126.2 | 1 | 0.681 | - | 13126.2 | - | - (stored whole) | 0.681 (whole/whole) | - | 6475.4 | 496.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | 4935.5 | 4935.5..4935.5 | 1 | 1.812 | - | 4935.5 | - | - (stored whole) | 1.812 (whole/whole) | - | 4205.8 | 544.0 | auc=0.500000, logloss=0.356515 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4128.2 | 4128.2..4128.2 | 1 | - | - | 4128.2 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 15380.3 | 15380.3..15380.3 | 1 | 0.268 | - | 15380.3 | - | - (stored whole) | 0.268 (whole/whole) | - | 1670.4 | 496.0 | auc=0.631766, logloss=0.528065 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 6565.0 | 6565.0..6565.0 | 1 | 0.629 | - | 6565.0 | - | - (stored whole) | 0.629 (whole/whole) | - | 1720.3 | 496.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | 1722.8 | 1722.8..1722.8 | 1 | 2.396 | - | 1722.8 | - | - (stored whole) | 2.396 (whole/whole) | - | 1314.0 | 544.0 | auc=0.500000, logloss=0.554692 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14659.0 | 14659.0..14659.0 | 1 | - | - | 14659.0 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 15195.6 | 15195.6..15195.6 | 1 | 0.965 | - | 15195.6 | - | - (stored whole) | 0.965 (whole/whole) | - | 5364.3 | 558.0 | accuracy=0.907780, mlogloss=0.258149 | yes | NOT-COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 33556.0 | 33556.0..33556.0 | 1 | 0.437 | - | 33556.0 | - | - (stored whole) | 0.437 (whole/whole) | - | 6693.0 | 558.0 | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-default-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | 170658.0 | 170658.0..170658.0 | 1 | 0.086 | - | 170658.0 | - | - (stored whole) | 0.086 (whole/whole) | - | 4394.5 | 936.0 | accuracy=0.911402, mlogloss=0.243459 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from release-board-resume-r2; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6926.4 | 6926.4..6926.4 | 1 | - | - | 6926.4 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 10221.5 | 10221.5..10221.5 | 1 | 0.678 | - | 10221.5 | - | - (stored whole) | 0.678 (whole/whole) | - | 1717.7 | 610.0 | accuracy=0.599270, mlogloss=1.012704 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 12711.7 | 12711.7..12711.7 | 1 | 0.545 | - | 12711.7 | - | - (stored whole) | 0.545 (whole/whole) | - | 1903.0 | 610.0 | accuracy=0.601200, mlogloss=1.005204 | yes | NOT-COMPARABLE | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | 76949.9 | 76949.9..76949.9 | 1 | 0.090 | - | 76949.9 | - | - (stored whole) | 0.090 (whole/whole) | - | 1457.7 | 856.0 | accuracy=0.601896, mlogloss=1.003173 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 36587.0 | 36587.0..36587.0 | 1 | - | - | 36587.0 | - | - (stored whole) | - | - | - | - | auc=0.979444, logloss=0.190928 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 39091.7 | 39091.7..39091.7 | 1 | 0.936 | - | 39091.7 | - | - (stored whole) | 0.936 (whole/whole) | - | 4881.6 | 430.0 | auc=0.979432, logloss=0.190474 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21192.5 | 21192.5..21192.5 | 1 | - | - | 21192.5 | - | - (stored whole) | - | - | - | - | auc=0.629203, logloss=0.529007 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 20288.2 | 20288.2..20288.2 | 1 | 1.045 | - | 20288.2 | - | - (stored whole) | 1.045 (whole/whole) | - | 1205.7 | 430.0 | auc=0.628945, logloss=0.528997 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6712.4 | 6712.4..6712.4 | 1 | - | - | 6712.4 | - | - (stored whole) | - | - | - | - | auc=0.980129, logloss=0.186686 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 8181.4 | 8181.4..8181.4 | 1 | 0.820 | - | 8181.4 | - | - (stored whole) | 0.820 (whole/whole) | - | 4878.1 | 428.0 | auc=0.980018, logloss=0.187292 | yes | UNKNOWN | - | ok (copied from opponents-default-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-default-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0320)`, ran on nv (RunPod L40S) job n0320

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3631.5 | 3631.5..3631.5 | 1 | - | - | 3631.5 | - | - (stored whole) | - | - | - | - | auc=0.630436, logloss=0.528554 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0320 2026-10-08; identity vs amd-mi325x: n/a) |
| catboost-gpu | catboost | gpu | opponent | 5982.0 | 5982.0..5982.0 | 1 | 0.607 | - | 5982.0 | - | - (stored whole) | 0.607 (whole/whole) | - | 1127.4 | 428.0 | auc=0.630310, logloss=0.528616 | yes | UNKNOWN | - | ok (copied from opponents-specific-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (copied from opponents-specific-20261006; measured this run) |

memory, ours, catboost-cpu: host not sampled; GPU not sampled

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on nv (RunPod L40S) job n0329

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 570.3 | 570.3..570.3 | 1 | - | - | 570.3 | - | - (stored whole) | - | - | - | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 509.0 | 509.0..509.0 | 1 | 1.120 | - | 1782.2 | 509.0 | 1273.19 (upload_ms_untimed) | 0.320 (whole/whole) | - | 4957.5 | 2154.0 | inertia=6.111e+17, n_iter=21 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu | torch | gpu | opponent | 849.0 | 849.0..849.0 | 1 | 0.672 | - | 924.5 | 849.0 | 75.47 (upload_ms_untimed) | 0.617 (whole/whole (kernel not derivable)) | - | 3162.6 | 3468.9 | inertia=5.991e+17, n_iter=55 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on nv (RunPod L40S) job n0329

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 203.7 | 203.7..203.7 | 1 | - | - | 203.7 | - | - (stored whole) | - | - | - | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 306.6 | 306.6..306.6 | 1 | 0.664 | - | 481.2 | 306.6 | 174.58 (upload_ms_untimed) | 0.423 (whole/whole) | - | 1227.4 | 614.0 | inertia=3.06e+08, n_iter=32 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu | torch | gpu | opponent | 609.7 | 609.7..609.7 | 1 | 0.334 | - | 618.9 | 609.7 | 9.25 (upload_ms_untimed) | 0.329 (whole/whole (kernel not derivable)) | - | 1251.1 | 438.0 | inertia=3.06e+08, n_iter=58 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0363)`, ran on nv (RunPod L40S) job n0363

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1029.2 | 1029.2..1029.2 | 1 | - | - | 1029.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0363 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 78.5 | 78.5..78.5 | 1 | 13.106 | - | 1328.2 | 78.5 | 1249.71 (upload_ms_untimed) | 0.775 (whole/whole) | - | 5075.9 | 2154.0 | finite=True, r2=-11031.855105, rmse=87.647429 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu | torch | gpu | opponent | 580.8 | 580.8..580.8 | 1 | 1.772 | - | 733.1 | 580.8 | 152.30 (upload_ms_untimed) | 1.404 (whole/whole (kernel not derivable)) | - | 3013.5 | 8894.4 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 19.8 | 19.8..19.8 | 1 | 51.878 | - | 159.6 | 19.8 | 139.80 (upload_ms_untimed) | 6.447 (whole/whole (kernel not derivable)) | - | 3044.8 | 3455.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0363)`, ran on nv (RunPod L40S) job n0363

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17.5 | 17.5..17.5 | 1 | - | - | 17.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908836, rmse=4.696479 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0363 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 22.1 | 22.1..22.1 | 1 | 0.794 | - | 196.3 | 22.1 | 174.26 (upload_ms_untimed) | 0.089 (whole/whole) | - | 1356.3 | 614.0 | finite=True, r2=0.908836, rmse=4.696488 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu | torch | gpu | opponent | 39.0 | 39.0..39.0 | 1 | 0.449 | - | 53.6 | 39.0 | 14.66 (upload_ms_untimed) | 0.326 (whole/whole (kernel not derivable)) | - | 1081.6 | 4649.6 | finite=True, r2=0.908836, rmse=4.696480 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | 8.961 | - | 16.2 | 2.0 | 14.27 (upload_ms_untimed) | 1.079 (whole/whole (kernel not derivable)) | - | 1098.7 | 376.4 | finite=True, r2=0.908836, rmse=4.696490 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical, wave 2

### gmm / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on nv (RunPod L40S) job n0329

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 571.5 | 571.5..571.5 | 1 | - | - | 571.5 | - | - (stored whole) | - | - | - | - | bic=-3.851e+07, mean_log_likelihood=200.794500, n_iter=24 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on nv (RunPod L40S) job n0329

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 33.1 | 33.1..33.1 | 1 | - | - | 33.1 | - | - (stored whole) | - | - | - | - | bic=-3.668e+06, mean_log_likelihood=12.807640, n_iter=29 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0566)`, ran on nv (RunPod L40S) job n0566

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1123.2 | 1123.2..1123.2 | 1 | - | - | 1123.2 | - | - (stored whole) | - | - | - | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0566 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuvs-gpu | cuvs | gpu | opponent | 301.6 | 301.6..301.6 | 1 | 3.723 | - | 635.6 | 301.6 | 333.95 (upload_ms_untimed) | 1.767 (whole/whole) | - | 1730.7 | 770.0 | recall_at_k=0.999975, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0566)`, ran on nv (RunPod L40S) job n0566

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 155.9 | 155.9..155.9 | 1 | - | - | 155.9 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999675, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0566 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuvs-gpu | cuvs | gpu | opponent | 297.5 | 297.5..297.5 | 1 | 0.524 | - | 323.7 | 297.5 | 26.15 (upload_ms_untimed) | 0.482 (whole/whole) | - | 933.3 | 450.0 | recall_at_k=0.999450, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on nv (RunPod L40S) job n0329

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1520.2 | 1520.2..1520.2 | 1 | - | - | 1520.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328682, rmse=0.684423 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 48.2 | 48.2..48.2 | 1 | 31.517 | - | 754.2 | 48.2 | 705.96 (upload_ms_untimed) | 2.016 (whole/whole) | - | 2969.7 | 1410.0 | finite=True, r2=-0.251259, rmse=0.934403 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-grid-logs.txt (nv/n0329)`, ran on nv (RunPod L40S) job n0329

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.5 | 4.5..4.5 | 1 | - | - | 4.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0329 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 15.2 | 15.2..15.2 | 1 | 0.294 | - | 71.7 | 15.2 | 56.48 (upload_ms_untimed) | 0.062 (whole/whole) | - | 1000.4 | 534.0 | finite=True, r2=0.908983, rmse=4.805051 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on nv (RunPod L40S) job n0526

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17.3 | 17.3..17.3 | 1 | - | - | 17.3 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-adafactor-synthetic-def/adafactor-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 15.3 | 15.3..15.3 | 1 | 1.137 | - | - | 15.3 | - (stored kernel) | 1.137 (MIXED ours whole / arm kernel) | - | 1035.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 169.7 | 169.7..169.7 | 1 | 0.102 | - | - | 169.7 | - (stored kernel) | 0.102 (MIXED ours whole / arm kernel) | - | 1118.9 | 1024.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adagrad / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on nv (RunPod L40S) job n0526

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10.8 | 10.8..10.8 | 1 | - | - | 10.8 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-adagrad-synthetic-def/adagrad-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 14.9 | 14.9..14.9 | 1 | 0.729 | - | - | 14.9 | - (stored kernel) | 0.729 (MIXED ours whole / arm kernel) | - | 968.5 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 115.6 | 115.6..115.6 | 1 | 0.094 | - | - | 115.6 | - (stored kernel) | 0.094 (MIXED ours whole / arm kernel) | - | 1302.1 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on nv (RunPod L40S) job n0526

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.9 | 13.9..13.9 | 1 | - | - | 13.9 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-adamax-synthetic-def/adamax-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 20.2 | 20.2..20.2 | 1 | 0.689 | - | - | 20.2 | - (stored kernel) | 0.689 (MIXED ours whole / arm kernel) | - | 981.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 146.9 | 146.9..146.9 | 1 | 0.095 | - | - | 146.9 | - (stored kernel) | 0.095 (MIXED ours whole / arm kernel) | - | 1445.6 | 896.0 | rel_fro_vs_torch_eager_fp32=6.395e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 2.5 | 2.5..2.5 | 1 | 0.061 | - | 716.0 | 2.5 | 713.58 (upload_ms_untimed) | 0.0002077 (whole/whole) | - | 3009.1 | 1440.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 38.0 | 38.0..38.0 | 1 | 0.004 | - | 38.0 | 38.0 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1499.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 2.4 | 2.4..2.4 | 1 | 0.052 | - | 59.7 | 2.4 | 57.33 (upload_ms_untimed) | 0.002 (whole/whole) | - | 892.9 | 486.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 0.058 | - | 2.2 | 2.2 | 0.00 (cpu-arm) | 0.058 (whole/whole) | - | 267.4 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on nv (RunPod L40S) job n0519

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 296.2 | 296.2..296.2 | 1 | - | - | 296.2 | - | - (stored whole) | - | - | - | - | relative_residual=2.9e-07 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs amd-mi325x: n/a) |
| torch-gpu | torch | gpu | opponent | 14.0 | 14.0..14.0 | 1 | 21.150 | - | - | 14.0 | - (stored kernel) | 21.150 (MIXED ours whole / arm kernel) | - | 1214.5 | 768.3 | relative_residual=1.509e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 14.9 | 14.9..14.9 | 1 | 19.872 | - | - | 14.9 | - (stored kernel) | 19.872 (MIXED ours whole / arm kernel) | - | 1285.6 | 1234.0 | relative_residual=1.365e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 4494.5 | 4494.5..4494.5 | 1 | 0.066 | - | 4494.5 | 4494.5 | 0.00 (cpu-arm) | 0.066 (whole/whole) | - | 2677.2 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on nv (RunPod L40S) job n0532

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 161.9 | 161.9..161.9 | 1 | - | - | 161.9 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559491 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 61.1 | 61.1..61.1 | 1 | 2.651 | - | 1467.5 | 61.1 | 1406.40 (upload_ms_untimed) | 0.110 (whole/whole) | - | 4710.5 | 1946.0 | accuracy=0.983067, logloss=0.559490 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 410.6 | 410.6..410.6 | 1 | 0.394 | - | 410.6 | 410.6 | 0.00 (cpu-arm) | 0.394 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on nv (RunPod L40S) job n0523

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.8 | 0.8..0.8 | 1 | - | - | 0.8 | - | - (stored whole) | - | - | - | - | n_components=81 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | opponent | 19.4 | 19.4..19.4 | 1 | 0.042 | - | 19.4 | 19.4 | 0.02 (upload_ms_untimed) | 0.042 (whole/whole) | - | 959.5 | 432.0 | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| networkx-cpu | networkx | cpu | opponent | 13.2 | 13.2..13.2 | 1 | 0.062 | - | 13.2 | 13.2 | 0.00 (cpu-arm) | 0.062 (whole/whole) | - | 117.1 | - | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on nv (RunPod L40S) job n0523

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.6 | 1.6..1.6 | 1 | - | - | 1.6 | - | - (stored whole) | - | - | - | - | n_components=588 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | opponent | 23.7 | 23.7..23.7 | 1 | 0.068 | - | 23.7 | 23.7 | 0.01 (upload_ms_untimed) | 0.068 (whole/whole) | - | 934.4 | 432.0 | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| networkx-cpu | networkx | cpu | opponent | 18.7 | 18.7..18.7 | 1 | 0.086 | - | 18.7 | 18.7 | 0.00 (cpu-arm) | 0.086 (whole/whole) | - | 99.0 | - | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on nv (RunPod L40S) job n0519

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21119.9 | 21119.9..21119.9 | 1 | - | - | 21119.9 | - | - (stored whole) | - | - | - | - | max_eigenvalue_error=5.95e-05, relative_residual=5.251e-05 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs amd-mi325x: n/a) |
| torch-gpu | torch | gpu | opponent | 81.2 | 81.2..81.2 | 1 | 260.080 | - | - | 81.2 | - (stored kernel) | 260.080 (MIXED ours whole / arm kernel) | - | 937.9 | 450.1 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 80.4 | 80.4..80.4 | 1 | 262.523 | - | - | 80.4 | - (stored kernel) | 262.523 (MIXED ours whole / arm kernel) | - | 753.9 | 1114.0 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 14605.9 | 14605.9..14605.9 | 1 | 1.446 | - | 14605.9 | 14605.9 | 0.00 (cpu-arm) | 1.446 (whole/whole) | - | 945.8 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.3 | 54.3..54.3 | 1 | - | - | 54.3 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.680693 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 31.4 | 31.4..31.4 | 1 | 1.729 | - | 31.4 | - | - (stored whole) | 1.729 (whole/whole) | - | 1813.0 | 438.0 | mean_abs_distortion=0.443920 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 32.3 | 32.3..32.3 | 1 | 1.684 | - | 32.3 | 32.3 | 0.00 (cpu-arm) | 1.684 (whole/whole) | - | 1134.6 | - | mean_abs_distortion=0.177966 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.345752 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 2.3 | 2.3..2.3 | 1 | 1.498 | - | 2.3 | - | - (stored whole) | 1.498 (whole/whole) | - | 894.4 | 438.0 | mean_abs_distortion=0.302259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.6 | 3.6..3.6 | 1 | 0.957 | - | 3.6 | 3.6 | 0.00 (cpu-arm) | 0.957 (whole/whole) | - | 257.1 | - | mean_abs_distortion=0.339791 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### incremental-pca / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 864.1 | 864.1..864.1 | 1 | - | - | 864.1 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=1.000000 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 1515.2 | 1515.2..1515.2 | 1 | 0.570 | - | 2206.1 | 1515.2 | 690.93 (upload_ms_untimed) | 0.392 (whole/whole) | - | 3076.8 | 1314.0 | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 17717.5 | 17717.5..17717.5 | 1 | 0.049 | - | 17717.5 | 17717.5 | 0.00 (cpu-arm) | 0.049 (whole/whole) | - | 2303.9 | - | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### incremental-pca / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 107.6 | 107.6..107.6 | 1 | - | - | 107.6 | - | - (stored whole) | - | - | - | - | explained_variance_fraction=0.999995 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 178.2 | 178.2..178.2 | 1 | 0.604 | - | 231.5 | 178.2 | 53.32 (upload_ms_untimed) | 0.465 (whole/whole) | - | 1176.1 | 518.0 | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 669.5 | 669.5..669.5 | 1 | 0.161 | - | 669.5 | 669.5 | 0.00 (cpu-arm) | 0.161 (whole/whole) | - | 307.8 | - | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0524)`, ran on nv (RunPod L40S) job n0524

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2328.3 | 2328.3..2328.3 | 1 | - | - | 2328.3 | - | - (stored whole) | - | - | - | - | recall_at_10=0.553500 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0524 2026-10-08; identity vs amd-mi325x: n/a) |
| cuvs-gpu | cuvs | gpu | opponent | 1905.2 | 1905.2..1905.2 | 1 | 1.222 | - | 2248.5 | 1905.2 | 343.31 (upload_ms_untimed) | 1.035 (whole/whole) | - | 1793.1 | 822.0 | recall_at_10=0.791300 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| faiss-cpu | faiss | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

memory, ours, faiss-cpu: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0524)`, ran on nv (RunPod L40S) job n0524

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 300.6 | 300.6..300.6 | 1 | - | - | 300.6 | - | - (stored whole) | - | - | - | - | recall_at_10=0.966550 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0524 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuvs-gpu | cuvs | gpu | opponent | 471.5 | 471.5..471.5 | 1 | 0.638 | - | 502.2 | 471.5 | 30.72 (upload_ms_untimed) | 0.599 (whole/whole) | - | 995.4 | 466.0 | recall_at_10=0.976875 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | opponent | 67512.8 | 67512.8..67512.8 | 1 | 0.004 | - | 67512.8 | 67512.8 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 135.3 | - | recall_at_10=0.979450 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2186.4 | 2186.4..2186.4 | 1 | - | - | 2186.4 | - | - (stored whole) | - | - | - | - | recall_at_10=0.809175 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuvs-gpu | cuvs | gpu | opponent | 1255.0 | 1255.0..1255.0 | 1 | 1.742 | - | 1596.3 | 1255.0 | 341.25 (upload_ms_untimed) | 1.370 (whole/whole) | - | 1815.7 | 822.0 | recall_at_10=0.993625 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| faiss-cpu | faiss | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (copied from release-board-resume-r2; measured this run) |

memory, ours, faiss-cpu: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-refine / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 349.6 | 349.6..349.6 | 1 | - | - | 349.6 | - | - (stored whole) | - | - | - | - | recall_at_10=0.999675 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuvs-gpu | cuvs | gpu | opponent | 381.6 | 381.6..381.6 | 1 | 0.916 | - | 408.0 | 381.6 | 26.37 (upload_ms_untimed) | 0.857 (whole/whole) | - | 1026.7 | 468.0 | recall_at_10=0.999050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | opponent | 70811.7 | 70811.7..70811.7 | 1 | 0.005 | - | 70811.7 | 70811.7 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 153.2 | - | recall_at_10=0.999375 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-sq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0318)`, ran on nv (RunPod L40S) job n0318

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 155.5 | 155.5..155.5 | 1 | - | - | 155.5 | - | - (stored whole) | - | - | - | - | recall_at_10=0.934975 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0318 2026-10-08; identity vs amd-mi325x: n/a) |
| cuvs-gpu | cuvs | gpu | opponent | 138.0 | 138.0..138.0 | 1 | 1.126 | - | 163.5 | 138.0 | 25.50 (upload_ms_untimed) | 0.951 (whole/whole) | - | 960.6 | 464.0 | recall_at_10=0.772500 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| faiss-cpu | faiss | cpu | opponent | 4836.7 | 4836.7..4836.7 | 1 | 0.032 | - | 4836.7 | 4836.7 | 0.00 (cpu-arm) | 0.032 (whole/whole) | - | 124.2 | - | recall_at_10=0.857050 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 235.2 | 235.2..235.2 | 1 | - | - | 235.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 739.0 | 739.0..739.0 | 1 | 0.318 | - | 1447.6 | 739.0 | 708.65 (upload_ms_untimed) | 0.162 (whole/whole) | - | 3068.2 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 5641.8 | 5641.8..5641.8 | 1 | 0.042 | - | 5641.8 | 5641.8 | 0.00 (cpu-arm) | 0.042 (whole/whole) | - | 1459.8 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10.2 | 10.2..10.2 | 1 | - | - | 10.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 26.6 | 26.6..26.6 | 1 | 0.382 | - | 85.0 | 26.6 | 58.41 (upload_ms_untimed) | 0.120 (whole/whole) | - | 952.0 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 173.1 | 173.1..173.1 | 1 | 0.059 | - | 173.1 | 173.1 | 0.00 (cpu-arm) | 0.059 (whole/whole) | - | 263.6 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on nv (RunPod L40S) job n0583

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5302.1 | 5302.1..5302.1 | 1 | - | - | 5302.1 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 15329.7 | 15329.7..15329.7 | 1 | 0.346 | - | 15334.1 | 15329.7 | 4.41 (upload_ms_untimed) | 0.346 (whole/whole) | - | 2042.9 | 438.0 | rel_error_vs_exact=0.032177 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | opponent | 25374.0 | 25374.0..25374.0 | 1 | 0.209 | - | 25374.0 | 25374.0 | 0.00 (cpu-arm) | 0.209 (whole/whole) | - | 2183.9 | - | rel_error_vs_exact=8.574e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on nv (RunPod L40S) job n0583

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 319.5 | 319.5..319.5 | 1 | - | - | 319.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.056e-07 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 595.0 | 595.0..595.0 | 1 | 0.537 | - | 599.4 | 595.0 | 4.34 (upload_ms_untimed) | 0.533 (whole/whole) | - | 1034.4 | 438.0 | rel_error_vs_exact=7.647e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| shap-cpu | shap | cpu | opponent | 9341.1 | 9341.1..9341.1 | 1 | 0.034 | - | 9341.1 | 9341.1 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 397.1 | - | rel_error_vs_exact=8.154e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0524)`, ran on nv (RunPod L40S) job n0524

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 67.4 | 67.4..67.4 | 1 | - | - | 67.4 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0524 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 23.2 | 23.2..23.2 | 1 | 2.908 | - | 23.2 | 23.2 | 0.00 (cpu-arm) | 2.908 (whole/whole) | - | 3499.1 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0524)`, ran on nv (RunPod L40S) job n0524

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.6 | 7.6..7.6 | 1 | - | - | 7.6 | - | - (stored whole) | - | - | - | - | masked_rmse=6.151696 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0524 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 6.0 | 6.0..6.0 | 1 | 1.267 | - | 6.0 | 6.0 | 0.00 (cpu-arm) | 1.267 (whole/whole) | - | 1952.7 | - | masked_rmse=5.263919 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 78.7 | 78.7..78.7 | 1 | - | - | 78.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x16 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 18.9 | 18.9..18.9 | 1 | 4.156 | - | 60.1 | 18.9 | 41.15 (upload_ms_untimed) | 1.310 (whole/whole) | - | 1423.1 | 558.0 | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 92.9 | 92.9..92.9 | 1 | 0.847 | - | 92.9 | 92.9 | 0.00 (cpu-arm) | 0.847 (whole/whole) | - | 672.8 | - | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1302.2 | 1302.2..1302.2 | 1 | - | - | 1302.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x259 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 56.3 | 56.3..56.3 | 1 | 23.149 | - | 99.2 | 56.3 | 42.99 (upload_ms_untimed) | 13.122 (whole/whole) | - | 3476.3 | 1560.0 | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 182.8 | 182.8..182.8 | 1 | 7.123 | - | 182.8 | 182.8 | 0.00 (cpu-arm) | 7.123 (whole/whole) | - | 6590.6 | - | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.4 | 8.4..8.4 | 1 | - | - | 8.4 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 7.9 | 7.9..7.9 | 1 | 1.062 | - | 37.3 | 7.9 | 29.43 (upload_ms_untimed) | 0.225 (whole/whole) | - | 1022.1 | 490.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 28.5 | 28.5..28.5 | 1 | 0.296 | - | 28.5 | 28.5 | 0.00 (cpu-arm) | 0.296 (whole/whole) | - | 308.0 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9.4 | 9.4..9.4 | 1 | - | - | 9.4 | - | - (stored whole) | - | - | - | - | output_shape=100000 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 38.9 | 38.9..38.9 | 1 | 0.242 | - | 68.5 | 38.9 | 29.58 (upload_ms_untimed) | 0.137 (whole/whole) | - | 1025.1 | 472.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 110.3 | 110.3..110.3 | 1 | 0.085 | - | 110.3 | 110.3 | 0.00 (cpu-arm) | 0.085 (whole/whole) | - | 295.4 | - | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lamb / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on nv (RunPod L40S) job n0526

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 23.8 | 23.8..23.8 | 1 | - | - | 23.8 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-lamb-synthetic-def/lamb-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 194.3 | 194.3..194.3 | 1 | - | - | 194.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.309043, rmse=0.694362 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 43.4 | 43.4..43.4 | 1 | 4.472 | - | 745.1 | 43.4 | 701.69 (upload_ms_untimed) | 0.261 (whole/whole) | - | 3008.6 | 1374.0 | finite=True, r2=0.328088, rmse=0.684726 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 922.6 | 922.6..922.6 | 1 | 0.211 | - | 922.6 | 922.6 | 0.00 (cpu-arm) | 0.211 (whole/whole) | - | 1975.3 | - | finite=True, r2=-4.245e+13, rmse=5.442e+06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 60.8 | 60.8..60.8 | 1 | - | - | 60.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 4.5 | 4.5..4.5 | 1 | 13.431 | - | 63.0 | 4.5 | 58.47 (upload_ms_untimed) | 0.965 (whole/whole) | - | 1046.3 | 498.0 | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 50.8 | 50.8..50.8 | 1 | 1.198 | - | 50.8 | 50.8 | 0.00 (cpu-arm) | 1.198 (whole/whole) | - | 292.2 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lion / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on nv (RunPod L40S) job n0526

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11.0 | 11.0..11.0 | 1 | - | - | 11.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-lion-synthetic-def/lion-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |

settings: {'betas': [0.9, 0.99], 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on nv (RunPod L40S) job n0523

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 178.8 | 178.8..178.8 | 1 | - | - | 178.8 | - | - (stored whole) | - | - | - | - | modularity=0.911187, n_communities=40 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | opponent | 218.4 | 218.4..218.4 | 1 | 0.819 | - | 218.4 | 218.4 | 0.01 (upload_ms_untimed) | 0.819 (whole/whole) | - | 947.9 | 440.0 | modularity=0.909430, n_communities=41 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| networkx-cpu | networkx | cpu | opponent | 2943.6 | 2943.6..2943.6 | 1 | 0.061 | - | 2943.6 | 2943.6 | 0.00 (cpu-arm) | 0.061 (whole/whole) | - | 239.0 | - | modularity=0.908460, n_communities=40 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on nv (RunPod L40S) job n0523

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 38.2 | 38.2..38.2 | 1 | - | - | 38.2 | - | - (stored whole) | - | - | - | - | modularity=0.941953, n_communities=58 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | opponent | 83.3 | 83.3..83.3 | 1 | 0.459 | - | 83.3 | 83.3 | 0.02 (upload_ms_untimed) | 0.458 (whole/whole) | - | 917.3 | 436.0 | modularity=0.941795, n_communities=62 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| networkx-cpu | networkx | cpu | opponent | 1444.9 | 1444.9..1444.9 | 1 | 0.026 | - | 1444.9 | 1444.9 | 0.00 (cpu-arm) | 0.026 (whole/whole) | - | 201.6 | - | modularity=0.940781, n_communities=56 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on nv (RunPod L40S) job n0521

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 413.7 | 413.7..413.7 | 1 | - | - | 413.7 | - | - (stored whole) | - | - | - | - | relative_residual=0.849956 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 220.0 | 220.0..220.0 | 1 | 1.880 | - | - | 220.0 | - (stored kernel) | 1.880 (MIXED ours whole / arm kernel) | - | 1783.9 | 3516.2 | relative_residual=nan | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 244.2 | 244.2..244.2 | 1 | 1.694 | - | - | 244.2 | - (stored kernel) | 1.694 (MIXED ours whole / arm kernel) | - | 2740.5 | 9158.0 | relative_residual=0.849957 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 7725.5 | 7725.5..7725.5 | 1 | 0.054 | - | 7725.5 | 7725.5 | 0.00 (cpu-arm) | 0.054 (whole/whole) | - | 4348.1 | - | relative_residual=0.876581 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on nv (RunPod L40S) job n0521

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12.1 | 12.1..12.1 | 1 | - | - | 12.1 | - | - (stored whole) | - | - | - | - | relative_residual=0.756366 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 7.1 | 7.1..7.1 | 1 | 1.712 | - | - | 7.1 | - (stored kernel) | 1.712 (MIXED ours whole / arm kernel) | - | 913.0 | 1122.8 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 9.9 | 9.9..9.9 | 1 | 1.222 | - | - | 9.9 | - (stored kernel) | 1.222 (MIXED ours whole / arm kernel) | - | 727.9 | 2778.0 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 183.1 | 183.1..183.1 | 1 | 0.066 | - | 183.1 | 183.1 | 0.00 (cpu-arm) | 0.066 (whole/whole) | - | 279.6 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on nv (RunPod L40S) job n0519

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1180.8 | 1180.8..1180.8 | 1 | - | - | 1180.8 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs amd-mi325x: n/a) |
| torch-gpu | torch | gpu | opponent | 36.1 | 36.1..36.1 | 1 | 32.689 | - | - | 36.1 | - (stored kernel) | 32.689 (MIXED ours whole / arm kernel) | - | 824.5 | 528.3 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 39.5 | 39.5..39.5 | 1 | 29.881 | - | - | 39.5 | - (stored kernel) | 29.881 (MIXED ours whole / arm kernel) | - | 875.6 | 1108.0 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| scipy-cpu | scipy | cpu | opponent | 5417.0 | 5417.0..5417.0 | 1 | 0.218 | - | 5417.0 | 5417.0 | 0.00 (cpu-arm) | 0.218 (whole/whole) | - | 600.1 | - | relative_residual=4.275e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, scipy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0519)`, ran on nv (RunPod L40S) job n0519

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 462.5 | 462.5..462.5 | 1 | - | - | 462.5 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0519 2026-10-08; identity vs amd-mi325x: n/a) |
| torch-gpu | torch | gpu | opponent | 36.1 | 36.1..36.1 | 1 | 12.808 | - | - | 36.1 | - (stored kernel) | 12.808 (MIXED ours whole / arm kernel) | - | 824.1 | 528.3 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 35.8 | 35.8..35.8 | 1 | 12.907 | - | - | 35.8 | - (stored kernel) | 12.907 (MIXED ours whole / arm kernel) | - | 778.4 | 984.0 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 10448.5 | 10448.5..10448.5 | 1 | 0.044 | - | 10448.5 | 10448.5 | 0.00 (cpu-arm) | 0.044 (whole/whole) | - | 1360.8 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 59.6 | 59.6..59.6 | 1 | - | - | 59.6 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 32.4 | 32.4..32.4 | 1 | 1.841 | - | 750.7 | 32.4 | 718.27 (upload_ms_untimed) | 0.079 (whole/whole) | - | 3171.9 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 153.4 | 153.4..153.4 | 1 | 0.389 | - | 153.4 | 153.4 | 0.00 (cpu-arm) | 0.389 (whole/whole) | - | 2214.7 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.0 | 4.0..4.0 | 1 | - | - | 4.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 3.4 | 3.4..3.4 | 1 | 1.169 | - | 60.1 | 3.4 | 56.62 (upload_ms_untimed) | 0.067 (whole/whole) | - | 892.7 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 41.8 | 41.8..41.8 | 1 | 0.096 | - | 41.8 | 41.8 | 0.00 (cpu-arm) | 0.096 (whole/whole) | - | 301.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0532)`, ran on nv (RunPod L40S) job n0532

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56.2 | 56.2..56.2 | 1 | - | - | 56.2 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559529 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0532 2026-10-08; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 50.0 | 50.0..50.0 | 1 | 1.123 | - | 1390.0 | 50.0 | 1339.98 (upload_ms_untimed) | 0.040 (whole/whole) | - | 4711.5 | 1946.0 | accuracy=0.983067, logloss=0.559524 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 454.4 | 454.4..454.4 | 1 | 0.124 | - | 454.4 | 454.4 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 4414.9 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nadam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on nv (RunPod L40S) job n0526

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.0 | 13.0..13.0 | 1 | - | - | 13.0 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-nadam-synthetic-def/nadam-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 20.8 | 20.8..20.8 | 1 | 0.625 | - | - | 20.8 | - (stored kernel) | 0.625 (MIXED ours whole / arm kernel) | - | 971.0 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 174.5 | 174.5..174.5 | 1 | 0.074 | - | - | 174.5 | - (stored kernel) | 0.074 (MIXED ours whole / arm kernel) | - | 1443.1 | 896.0 | rel_fro_vs_torch_eager_fp32=1.498e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.2 | 0.2..0.2 | 1 | - | - | 0.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 2.2 | 2.2..2.2 | 1 | 0.078 | - | 705.7 | 2.2 | 703.55 (upload_ms_untimed) | 0.0002393 (whole/whole) | - | 3051.7 | 1450.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 39.2 | 39.2..39.2 | 1 | 0.004 | - | 39.2 | 39.2 | 0.00 (cpu-arm) | 0.004 (whole/whole) | - | 1459.3 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | 0.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 0.6 | 0.6..0.6 | 1 | 0.175 | - | 54.9 | 0.6 | 54.32 (upload_ms_untimed) | 0.002 (whole/whole) | - | 935.1 | 496.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.5 | 2.5..2.5 | 1 | 0.043 | - | 2.5 | 2.5 | 0.00 (cpu-arm) | 0.043 (whole/whole) | - | 265.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 66.5 | 66.5..66.5 | 1 | - | - | 66.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x119 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 52.0 | 52.0..52.0 | 1 | 1.279 | - | 82.4 | 52.0 | 30.41 (upload_ms_untimed) | 0.807 (whole/whole) | - | 1251.6 | 524.0 | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 93.8 | 93.8..93.8 | 1 | 0.708 | - | 93.8 | 93.8 | 0.00 (cpu-arm) | 0.708 (whole/whole) | - | 531.2 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 50.1 | 50.1..50.1 | 1 | - | - | 50.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x508 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 173.9 | 173.9..173.9 | 1 | 0.288 | - | 202.4 | 173.9 | 28.46 (upload_ms_untimed) | 0.248 (whole/whole) | - | 1557.8 | 656.0 | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 48.4 | 48.4..48.4 | 1 | 1.035 | - | 48.4 | 48.4 | 0.00 (cpu-arm) | 1.035 (whole/whole) | - | 1399.8 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on nv (RunPod L40S) job n0523

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.4 | 1.4..1.4 | 1 | - | - | 1.4 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | opponent | 5.0 | 5.0..5.0 | 1 | 0.284 | - | 5.0 | 5.0 | 0.01 (upload_ms_untimed) | 0.283 (whole/whole) | - | 1009.8 | 440.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| networkx-cpu | networkx | cpu | opponent | 274.0 | 274.0..274.0 | 1 | 0.005 | - | 274.0 | 274.0 | 0.00 (cpu-arm) | 0.005 (whole/whole) | - | 211.2 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0523)`, ran on nv (RunPod L40S) job n0523

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.3 | 1.3..1.3 | 1 | - | - | 1.3 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0523 2026-10-08; identity vs amd-mi325x: MATCH) |
| cugraph-gpu | cugraph | gpu | opponent | 5.0 | 5.0..5.0 | 1 | 0.257 | - | 5.0 | 5.0 | 0.02 (upload_ms_untimed) | 0.256 (whole/whole) | - | 1023.2 | 438.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| networkx-cpu | networkx | cpu | opponent | 189.4 | 189.4..189.4 | 1 | 0.007 | - | 189.4 | 189.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 175.9 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on nv (RunPod L40S) job n0583

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7393.5 | 7393.5..7393.5 | 1 | - | - | 7393.5 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 2320.3 | 2320.3..2320.3 | 1 | 3.186 | - | 2324.4 | 2320.3 | 4.11 (upload_ms_untimed) | 3.181 (whole/whole) | - | 1915.3 | 438.0 | rel_error_vs_exact=1.483e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | opponent | 90631.6 | 90631.6..90631.6 | 1 | 0.082 | - | 90631.6 | 90631.6 | 0.00 (cpu-arm) | 0.082 (whole/whole) | - | 1725.8 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0583)`, ran on nv (RunPod L40S) job n0583

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 98.0 | 98.0..98.0 | 1 | - | - | 98.0 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@03b648834 nv/n0583 2026-10-09; identity vs amd-mi325x: MATCH) |
| cuml-gpu | cuml | gpu | opponent | 319.6 | 319.6..319.6 | 1 | 0.306 | - | 323.9 | 319.6 | 4.28 (upload_ms_untimed) | 0.302 (whole/whole) | - | 956.8 | 438.0 | rel_error_vs_exact=1.701e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| shap-cpu | shap | cpu | opponent | 133.5 | 133.5..133.5 | 1 | 0.733 | - | 133.5 | 133.5 | 0.00 (cpu-arm) | 0.733 (whole/whole) | - | 476.7 | - | rel_error_vs_exact=1.144e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.5 | 0.5..0.5 | 1 | - | - | 0.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x152 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 1.0 | 1.0..1.0 | 1 | 0.488 | - | 65.0 | 1.0 | 64.00 (upload_ms_untimed) | 0.008 (whole/whole) | - | 1945.9 | 560.0 | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.2 | 3.2..3.2 | 1 | 0.156 | - | 3.2 | 3.2 | 0.00 (cpu-arm) | 0.156 (whole/whole) | - | 1430.3 | - | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | 0.3 | - | - (stored whole) | - | - | - | - | output_shape=100000x77 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 1.5 | 1.5..1.5 | 1 | 0.186 | - | 72.6 | 1.5 | 71.09 (upload_ms_untimed) | 0.004 (whole/whole) | - | 944.4 | 510.0 | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 0.125 | - | 2.2 | 2.2 | 0.00 (cpu-arm) | 0.125 (whole/whole) | - | 369.8 | - | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0318)`, ran on nv (RunPod L40S) job n0318

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 120.9 | 120.9..120.9 | 1 | - | - | 120.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0318 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 3161.1 | 3161.1..3161.1 | 1 | 0.038 | - | 3215.7 | 3161.1 | 54.57 (upload_ms_untimed) | 0.038 (whole/whole) | - | 942.6 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 4294.8 | 4294.8..4294.8 | 1 | 0.028 | - | 4294.8 | 4294.8 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 397.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on nv (RunPod L40S) job n0521

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1177.1 | 1177.1..1177.1 | 1 | - | - | 1177.1 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.485e-07 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 227.1 | 227.1..227.1 | 1 | 5.183 | - | - | 227.1 | - (stored kernel) | 5.183 (MIXED ours whole / arm kernel) | - | 1703.4 | 3008.9 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 230.5 | 230.5..230.5 | 1 | 5.106 | - | - | 230.5 | - (stored kernel) | 5.106 (MIXED ours whole / arm kernel) | - | 2486.4 | 5160.0 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 17211.6 | 17211.6..17211.6 | 1 | 0.068 | - | 17211.6 | 17211.6 | 0.00 (cpu-arm) | 0.068 (whole/whole) | - | 8528.1 | - | relative_gram_difference=2.459e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on nv (RunPod L40S) job n0521

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 57.2 | 57.2..57.2 | 1 | - | - | 57.2 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.714e-07 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 7.0 | 7.0..7.0 | 1 | 8.190 | - | - | 7.0 | - (stored kernel) | 8.190 (MIXED ours whole / arm kernel) | - | 831.1 | 168.1 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 6.8 | 6.8..6.8 | 1 | 8.357 | - | - | 6.8 | - (stored kernel) | 8.357 (MIXED ours whole / arm kernel) | - | 653.7 | 726.0 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 444.9 | 444.9..444.9 | 1 | 0.129 | - | 444.9 | 444.9 | 0.00 (cpu-arm) | 0.129 (whole/whole) | - | 475.9 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0328)`, ran on nv (RunPod L40S) job n0328

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 358.7 | 358.7..358.7 | 1 | - | - | 358.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.039998, rmse=0.851877 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0328 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.404e+06 | 1.404e+06..1.404e+06 | 1 | 0.0002554 | - | 1.404e+06 | 1.404e+06 | 0.00 (cpu-arm) | 0.0002554 (whole/whole) | - | 7995.7 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0328)`, ran on nv (RunPod L40S) job n0328

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 234.9 | 234.9..234.9 | 1 | - | - | 234.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.899596, rmse=5.046749 | - | main board, one scored run | - | ok (main@de2b2b739 nv/n0328 2026-10-08; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 386298.7 | 386298.7..386298.7 | 1 | 0.0006081 | - | 386298.7 | 386298.7 | 0.00 (cpu-arm) | 0.0006081 (whole/whole) | - | 926.3 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 783.2 | 783.2..783.2 | 1 | - | - | 783.2 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.0002359 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 65.7 | 65.7..65.7 | 1 | 11.927 | - | - | 65.7 | - (stored kernel) | 11.927 (MIXED ours whole / arm kernel) | - | 1637.5 | 1012.2 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1589.1 | 1589.1..1589.1 | 1 | 0.493 | - | 1589.1 | 1589.1 | 0.00 (cpu-arm) | 0.493 (whole/whole) | - | 2079.3 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 113.1 | 113.1..113.1 | 1 | - | - | 113.1 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.027197 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 32.0 | 32.0..32.0 | 1 | 3.537 | - | - | 32.0 | - (stored kernel) | 3.537 (MIXED ours whole / arm kernel) | - | 838.9 | 198.1 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1017.5 | 1017.5..1017.5 | 1 | 0.111 | - | 1017.5 | 1017.5 | 0.00 (cpu-arm) | 0.111 (whole/whole) | - | 477.8 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0004)`, ran on nv (RunPod L40S) job n0004

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5849.8 | 5849.8..5849.8 | 1 | - | - | 5849.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@8d8771a8e nv/n0004 2026-10-07; identity vs amd-mi325x: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2252.8 | 2252.8..2252.8 | 1 | 2.597 | - | 2252.8 | 2252.8 | 0.00 (cpu-arm) | 2.597 (whole/whole) | - | 378.4 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0526)`, ran on nv (RunPod L40S) job n0526

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10.6 | 10.6..10.6 | 1 | - | - | 10.6 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/n0526/work-rmsprop-synthetic-def/rmsprop-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0526 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 13.3 | 13.3..13.3 | 1 | 0.803 | - | - | 13.3 | - (stored kernel) | 0.803 (MIXED ours whole / arm kernel) | - | 958.5 | 896.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 118.3 | 118.3..118.3 | 1 | 0.090 | - | - | 118.3 | - (stored kernel) | 0.090 (MIXED ours whole / arm kernel) | - | 1308.8 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 232.2 | 232.2..232.2 | 1 | - | - | 232.2 | - | - (stored whole) | - | - | - | - | output_shape=100000x220 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 828.4 | 828.4..828.4 | 1 | 0.280 | - | 1526.7 | 828.4 | 698.31 (upload_ms_untimed) | 0.152 (whole/whole) | - | 3064.3 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 6394.7 | 6394.7..6394.7 | 1 | 0.036 | - | 6394.7 | 6394.7 | 0.00 (cpu-arm) | 0.036 (whole/whole) | - | 1455.4 | - | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10.0 | 10.0..10.0 | 1 | - | - | 10.0 | - | - (stored whole) | - | - | - | - | output_shape=100000x11 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 34.3 | 34.3..34.3 | 1 | 0.291 | - | 91.2 | 34.3 | 56.92 (upload_ms_untimed) | 0.109 (whole/whole) | - | 947.6 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 184.7 | 184.7..184.7 | 1 | 0.054 | - | 184.7 | 184.7 | 0.00 (cpu-arm) | 0.054 (whole/whole) | - | 265.1 | - | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1561.5 | 1561.5..1561.5 | 1 | - | - | 1561.5 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.327829, rmse=0.684858 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 4857.9 | 4857.9..4857.9 | 1 | 0.321 | - | 5558.3 | 4857.9 | 700.38 (upload_ms_untimed) | 0.281 (whole/whole) | - | 2928.5 | 1374.0 | finite=True, r2=0.327768, rmse=0.684889 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 55283.2 | 55283.2..55283.2 | 1 | 0.028 | - | 55283.2 | 55283.2 | 0.00 (cpu-arm) | 0.028 (whole/whole) | - | 1133.1 | - | finite=True, r2=-2.197e+24, rmse=1.238e+12 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1024.3 | 1024.3..1024.3 | 1 | - | - | 1024.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908969, rmse=4.805411 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 1579.7 | 1579.7..1579.7 | 1 | 0.648 | - | 1642.0 | 1579.7 | 62.33 (upload_ms_untimed) | 0.624 (whole/whole) | - | 971.4 | 498.0 | finite=True, r2=0.908979, rmse=4.805168 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 13336.4 | 13336.4..13336.4 | 1 | 0.077 | - | 13336.4 | 13336.4 | 0.00 (cpu-arm) | 0.077 (whole/whole) | - | 259.8 | - | finite=True, r2=0.880681, rmse=5.501638 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### simple-imputer / istella (rows full, shape X 1000000x220; X_true 1000000x220; Xq 100000x220; Xq_true 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 240.2 | 240.2..240.2 | 1 | - | - | 240.2 | - | - (stored whole) | - | - | - | - | masked_rmse=346849.129968 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 250.6 | 250.6..250.6 | 1 | 0.959 | - | 981.6 | 250.6 | 731.04 (upload_ms_untimed) | 0.245 (whole/whole) | - | 4914.9 | 1442.0 | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 36684.7 | 36684.7..36684.7 | 1 | 0.007 | - | 36684.7 | 36684.7 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 8060.1 | - | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### simple-imputer / taxi (rows full, shape X 1000000x11; X_true 1000000x11; Xq 100000x11; Xq_true 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9.8 | 9.8..9.8 | 1 | - | - | 9.8 | - | - (stored whole) | - | - | - | - | masked_rmse=5.985180 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 22.4 | 22.4..22.4 | 1 | 0.440 | - | 84.8 | 22.4 | 62.39 (upload_ms_untimed) | 0.116 (whole/whole) | - | 1043.3 | 488.0 | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1480.4 | 1480.4..1480.4 | 1 | 0.007 | - | 1480.4 | 1480.4 | 0.00 (cpu-arm) | 0.007 (whole/whole) | - | 597.8 | - | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.2 | 54.2..54.2 | 1 | - | - | 54.2 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=1.883381 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 108.4 | 108.4..108.4 | 1 | 0.500 | - | 108.4 | - | - (stored whole) | 0.500 (whole/whole) | - | 1993.6 | 430.0 | mean_abs_distortion=0.876709 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 32.5 | 32.5..32.5 | 1 | 1.671 | - | 32.5 | 32.5 | 0.00 (cpu-arm) | 1.671 (whole/whole) | - | 1134.7 | - | mean_abs_distortion=0.474347 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sparse-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0314)`, ran on nv (RunPod L40S) job n0314

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.7 | 4.7..4.7 | 1 | - | - | 4.7 | - | - (stored whole) | - | - | - | - | mean_abs_distortion=0.147163 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0314 2026-10-08; identity vs amd-mi325x: n/a) |
| cuml-gpu | cuml | gpu | opponent | 5.9 | 5.9..5.9 | 1 | 0.794 | - | 5.9 | - | - (stored whole) | 0.794 (whole/whole) | - | 932.9 | 430.0 | mean_abs_distortion=0.266849 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.9 | 3.9..3.9 | 1 | 1.186 | - | 3.9 | 3.9 | 0.00 (cpu-arm) | 1.186 (whole/whole) | - | 257.7 | - | mean_abs_distortion=0.381016 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on nv (RunPod L40S) job n0521

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1516.7 | 1516.7..1516.7 | 1 | - | - | 1516.7 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=662.224271, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 266.0 | 266.0..266.0 | 1 | 5.703 | - | - | 266.0 | - (stored kernel) | 5.703 (MIXED ours whole / arm kernel) | - | 1883.3 | 5176.3 | max_rel_singular_value_error=8.068917, relative_reconstruction_error_100k_rows=2.861e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 242.8 | 242.8..242.8 | 1 | 6.246 | - | - | 242.8 | - (stored kernel) | 6.246 (MIXED ours whole / arm kernel) | - | 2666.2 | 9146.0 | max_rel_singular_value_error=10270.720886, relative_reconstruction_error_100k_rows=2.385e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 13448.4 | 13448.4..13448.4 | 1 | 0.113 | - | 13448.4 | 13448.4 | 0.00 (cpu-arm) | 0.113 (whole/whole) | - | 8679.1 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0521)`, ran on nv (RunPod L40S) job n0521

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 64.2 | 64.2..64.2 | 1 | - | - | 64.2 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | - | ok (main@4df610f3b nv/n0521 2026-10-08; identity vs amd-mi325x: MATCH) |
| torch-gpu | torch | gpu | opponent | 8.6 | 8.6..8.6 | 1 | 7.464 | - | - | 8.6 | - (stored kernel) | 7.464 (MIXED ours whole / arm kernel) | - | 850.4 | 1186.8 | max_rel_singular_value_error=2.595e-06, relative_reconstruction_error_100k_rows=7.244e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| cupy-gpu | cupy | gpu | opponent | 8.3 | 8.3..8.3 | 1 | 7.753 | - | - | 8.3 | - (stored kernel) | 7.753 (MIXED ours whole / arm kernel) | - | 658.5 | 2766.0 | max_rel_singular_value_error=3.763e-06, relative_reconstruction_error_100k_rows=6.886e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-default-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 319.0 | 319.0..319.0 | 1 | 0.201 | - | 319.0 | 319.0 | 0.00 (cpu-arm) | 0.201 (whole/whole) | - | 450.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0313)`, ran on nv (RunPod L40S) job n0313

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 347.3 | 347.3..347.3 | 1 | - | - | 347.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@89485bc96 nv/n0313 2026-10-08; identity vs amd-mi325x: MATCH) |
| gpytorch-gpu | gpytorch | gpu | opponent | 36.2 | 36.2..36.2 | 1 | 9.581 | - | - | 36.2 | - (stored kernel) | 9.581 (MIXED ours whole / arm kernel) | - | 2157.6 | 1492.0 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-specific-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 14121.1 | 14121.1..14121.1 | 1 | 0.025 | - | 14121.1 | 14121.1 | 0.00 (cpu-arm) | 0.025 (whole/whole) | - | 3161.9 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / istella (rows full, shape X 100000x220; Xq 10000x220; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0530)`, ran on nv (RunPod L40S) job n0530

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17.4 | 17.4..17.4 | 1 | - | - | 17.4 | - | - (stored whole) | - | - | - | - | max_additivity_error=1.175e-06 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0530 2026-10-08; identity vs amd-mi325x: MATCH) |
| xgboost-gpu | xgboost | gpu | opponent | 30.2 | 30.2..30.2 | 1 | 0.575 | - | 30.2 | - | - (stored whole) | 0.575 (whole/whole) | - | 1535.8 | 530.0 | max_additivity_error=1.956e-06 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-specific-20261006; measured this run) |
| shap-cpu | shap | cpu | opponent | 822.3 | 822.3..822.3 | 1 | 0.021 | - | 822.3 | 822.3 | 0.00 (cpu-arm) | 0.021 (whole/whole) | - | 1561.9 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 803.0 | 803.0..803.0 | 1 | 0.022 | - | 803.0 | 803.0 | 0.00 (cpu-arm) | 0.022 (whole/whole) | - | 1439.3 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 572.6 | 572.6..572.6 | 1 | 0.030 | - | 572.6 | 572.6 | 0.00 (cpu-arm) | 0.030 (whole/whole) | - | 1526.3 | - | max_additivity_error=4.441e-15 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / taxi (rows full, shape X 100000x11; Xq 10000x11; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/nv-results.txt (nv/n0530)`, ran on nv (RunPod L40S) job n0530

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.2 | 2.2..2.2 | 1 | - | - | 2.2 | - | - (stored whole) | - | - | - | - | max_additivity_error=3.858e-05 | - | main board, one scored run | - | ok (main@42d1e42c6 nv/n0530 2026-10-08; identity vs amd-mi325x: MATCH) |
| xgboost-gpu | xgboost | gpu | opponent | 51.2 | 51.2..51.2 | 1 | 0.042 | - | 51.2 | - | - (stored whole) | 0.042 (whole/whole) | - | 485.3 | 500.0 | max_additivity_error=5.402e-05 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from opponents-default-20261006; measured this run) |
| shap-cpu | shap | cpu | opponent | 357.4 | 357.4..357.4 | 1 | 0.006 | - | 357.4 | 357.4 | 0.00 (cpu-arm) | 0.006 (whole/whole) | - | 485.3 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 338.8 | 338.8..338.8 | 1 | 0.006 | - | 338.8 | 338.8 | 0.00 (cpu-arm) | 0.006 (whole/whole) | - | 374.0 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 234.4 | 234.4..234.4 | 1 | 0.009 | - | 234.4 | 234.4 | 0.00 (cpu-arm) | 0.009 (whole/whole) | - | 263.0 | - | max_additivity_error=5.684e-13 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

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

