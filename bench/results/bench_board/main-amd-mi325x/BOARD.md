# mojolearn benchmark board

Generated 2026-10-09T15:40:51Z from `board.json` (schema `mojolearn-bench-board/1`).

> MAIN BOARD amd-mi325x, version label main@8ed94710a. Unreleased: not reproducible by pip install; the release boards are the reference.

> Cells: 86 races; oldest cell main@89485bc96 (2026-10-08T13:27:06Z), newest cell main@8ed94710a (2026-10-09T04:43:28Z). Boxes: amd (DigitalOcean MI325X).

> Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an older ok cell stays, flagged with the newer failed run). 90 replaced or failed observations are in LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on this board.

> Ours: one scored run per cell (lq RACE ALGOS lines, lq CMD bench_board summaries); the status column names the cell's commit, box/job, commit date and the other vendor's digest at the same commit (identity: MATCH 42, n/a 44).

> Opponents: copied from the stored opponent boards (opponents-20261006, release-board-resume-r2), never re-run here; `ours IDENTICAL / arm` divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded upload_ms_separate. Opponents withheld for changed lane settings: 2 races.

## Identity

Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA and AMD). Counts: MATCH 42, n/a 44.

DIFFER: none.

## FAILED

Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board flagged with the failed run. 41 failed runs.

| lane | dataset | commit | box/job | reason | ok cell on the board |
|---|---|---|---|---|---|
| batchnorm2d | synthetic | main@42d1e42c6 | amd/a0883 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| batchnorm2d | synthetic | main@42d1e42c6 | amd/a0873 | not_ready | none |
| batchnorm2d | synthetic | main@530498ea8 | amd/a0836 | not_ready | none |
| cnn-clf | synthetic | main@89485bc96 | amd/a0497 | error | none |
| conv2d | synthetic | main@42d1e42c6 | amd/a0875 | not_ready | none |
| dropout2d | synthetic | main@42d1e42c6 | amd/a0883 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| dropout2d | synthetic | main@42d1e42c6 | amd/a0873 | not_ready | none |
| dropout2d | synthetic | main@530498ea8 | amd/a0836 | not_ready | none |
| embedding | synthetic | main@42d1e42c6 | amd/a0883 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| embedding | synthetic | main@42d1e42c6 | amd/a0873 | not_ready | none |
| embedding | synthetic | main@530498ea8 | amd/a0836 | not_ready | none |
| gcn | istella | main@4303e1bfb | amd/a0837 | not_ready | none |
| gcn | taxi | main@4303e1bfb | amd/a0837 | not_ready | none |
| graphsage | istella | main@4303e1bfb | amd/a0837 | not_ready | none |
| graphsage | taxi | main@4303e1bfb | amd/a0837 | not_ready | none |
| gru-clf | synthetic | main@89485bc96 | amd/a0498 | error | none |
| gru-clf | taxi-hourly | main@89485bc96 | amd/a0498 | error | none |
| gru-reg | synthetic | main@89485bc96 | amd/a0498 | error | none |
| gru-reg | taxi-hourly | main@89485bc96 | amd/a0498 | error | none |
| ivf-pq | istella | main@8ed94710a | amd/a1067 | timeout | none |
| ivf-pq | istella | main@4df610f3b | amd/a0871 | timeout | none |
| layernorm | synthetic | main@42d1e42c6 | amd/a0883 | REFUSED(not_ready:_{"error":_"ModuleNotFoundError(\"No_module_named_'torch'\")",_"event":_"error",_"stage":_"ready"}) | none |
| layernorm | synthetic | main@42d1e42c6 | amd/a0873 | not_ready | none |
| layernorm | synthetic | main@530498ea8 | amd/a0836 | not_ready | none |
| lstm-clf | synthetic | main@89485bc96 | amd/a0498 | error | none |
| lstm-clf | taxi-hourly | main@89485bc96 | amd/a0498 | error | none |
| lstm-reg | synthetic | main@89485bc96 | amd/a0498 | error | none |
| lstm-reg | taxi-hourly | main@89485bc96 | amd/a0498 | error | none |
| maxpool2d | synthetic | main@530498ea8 | amd/a0836 | not_ready | none |
| moe | synthetic | main@42d1e42c6 | amd/a0875 | not_ready | none |
| moe | synthetic | main@530498ea8 | amd/a0836 | not_ready | none |
| resnet-block | synthetic | main@42d1e42c6 | amd/a0875 | not_ready | none |
| resnet-block | synthetic | main@530498ea8 | amd/a0836 | not_ready | none |
| rnn-clf | synthetic | main@89485bc96 | amd/a0498 | error | none |
| rnn-clf | taxi-hourly | main@89485bc96 | amd/a0498 | error | none |
| rnn-reg | synthetic | main@89485bc96 | amd/a0498 | error | none |
| rnn-reg | taxi-hourly | main@89485bc96 | amd/a0498 | error | none |
| conv2d | synthetic | main@e5f3f2ed8 | amd/a0885 | NO-RECORD | none |
| moe | synthetic | main@e5f3f2ed8 | amd/a0885 | NO-RECORD | none |
| resnet-block | synthetic | main@e5f3f2ed8 | amd/a0885 | NO-RECORD | none |
| gbdt-categorical | istella | main@89485bc96 | amd/a0499 | NO-RECORD | none |

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
| mojolearn | main@8ed94710a (wheel none (unreleased; built from source at each cell's commit), sha256 -) |
| script commit | 8ed94710ac8ef06fb3a58e44b4c90416f269dda7 |
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

Races: 86 planned, 86 done, 0 failed, 0 unsupported, 0 pending. Cells: 219 (REFUSED 7, ok 212).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.603e-09 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.17e-09 |
| algos | cholesky | synthetic | relative_residual | - | 2.9e-07 | torch-gpu 1.044e-07; numpy-cpu 3.928e-08 |
| algos | complement-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | 0.559491 | sklearn-cpu 0.557285 |
| algos | connected-components | istella | n_components | - | 81 | networkx-cpu 81 |
| algos | connected-components | taxi | n_components | - | 588 | networkx-cpu 588 |
| algos | damped-ets | synthetic | forecast_rmse (lower is better) | - | 13.931153 | statsmodels-cpu 26.588738; statsforecast-cpu 13.945986 |
| algos | damped-ets | taxi-hourly | forecast_rmse (lower is better) | - | 96.690449 | statsmodels-cpu 196.955273; statsforecast-cpu 96.685568 |
| algos | eigh | synthetic | max_eigenvalue_error | - | 5.95e-05 | torch-gpu 1.254e-06; numpy-cpu 3.49e-08 |
| algos | eigh | synthetic | relative_residual | - | 5.251e-05 | torch-gpu 1.269e-06; numpy-cpu 2.824e-08 |
| algos | enet-cv | istella | r2 (higher is better) | - | 0.316583 | sklearn-cpu 0.326805 |
| algos | enet-cv | istella | rmse (lower is better) | - | 0.690563 | sklearn-cpu 0.685379 |
| algos | enet-cv | taxi | r2 (higher is better) | - | 0.909002 | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | - | 4.804540 | sklearn-cpu 4.804486 |
| algos | gaussian-nb | istella | accuracy (higher is better) | - | 0.876570 | sklearn-cpu 0.876530 |
| algos | gaussian-nb | istella | logloss (lower is better) | - | 3.574420 | sklearn-cpu 3.417392 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | 0.719820 | sklearn-cpu 0.719900 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | 1.132249 | sklearn-cpu 1.133898 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | 0.967575 | faiss-cpu 0.979450 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | 8.307e-08 | shap-cpu 8.63e-15 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | 1.056e-07 | shap-cpu 8.149e-15 |
| algos | knn-imputer | istella | masked_rmse | - | 323953.237332 | sklearn-cpu 986208.700423 |
| algos | knn-imputer | taxi | masked_rmse | - | 6.151696 | sklearn-cpu 5.263919 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.310329 | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.693715 | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909059 | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803051 | sklearn-cpu 4.803593 |
| algos | lda-clf | istella | accuracy (higher is better) | - | 0.912830 | sklearn-cpu 0.899520 |
| algos | lda-clf | istella | logloss (lower is better) | - | 0.235875 | sklearn-cpu 0.439049 |
| algos | lda-clf | taxi | accuracy (higher is better) | - | 0.762530 | sklearn-cpu 0.762530 |
| algos | lda-clf | taxi | logloss (lower is better) | - | 0.539763 | sklearn-cpu 0.539767 |
| algos | louvain | istella | modularity | - | 0.911187 | networkx-cpu 0.908460 |
| algos | louvain | istella | n_communities | - | 40 | networkx-cpu 40 |
| algos | louvain | taxi | modularity | - | 0.941953 | networkx-cpu 0.940781 |
| algos | louvain | taxi | n_communities | - | 58 | networkx-cpu 56 |
| algos | lstsq | istella | relative_residual | - | 0.849956 | torch-gpu nan; numpy-cpu 0.876106 |
| algos | lstsq | taxi | relative_residual | - | 0.756366 | torch-gpu 0.756366; numpy-cpu 0.756366 |
| algos | lu-factor | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 4.003e-07; scipy-cpu 3.439e-07 |
| algos | lu-solve | synthetic | relative_residual | - | 3.256e-06 | torch-gpu 4.041e-07; numpy-cpu 3.259e-08 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | 0.983067 | sklearn-cpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | 0.559529 | sklearn-cpu 0.557319 |
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
| algos | ridge-cv | istella | r2 (higher is better) | - | 0.328684 | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | 0.684422 | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | 0.908981 | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | 4.805109 | sklearn-cpu 4.805057 |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.375e-09 |
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
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | torch-gpu 5.991e+17; sklearn-cpu 6.049e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | torch-gpu -; sklearn-cpu 0.999759 |
| classical | kmeans | istella | n_iter | - | 33 | torch-gpu 91; sklearn-cpu 24 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | torch-gpu 3.06e+08; sklearn-cpu 3.093e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | torch-gpu -; sklearn-cpu 0.999937 |
| classical | kmeans | taxi | n_iter | - | 91 | torch-gpu 54; sklearn-cpu 58 |
| classical | ols | istella | r2 (higher is better) | - | 0.332506 | torch-gpu nan; torch-gpu-eigh 0.151604; sklearn-cpu 0.001881 |
| classical | ols | istella | rmse (lower is better) | - | 0.681740 | torch-gpu nan; torch-gpu-eigh 0.768589; sklearn-cpu 0.833655 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908836 | torch-gpu 0.908840; torch-gpu-eigh 0.908822; sklearn-cpu 0.724850 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696479 | torch-gpu 4.696376; torch-gpu-eigh 4.696849; sklearn-cpu 8.159187 |
| classical2 | gmm | istella | bic (lower is better) | - | -3.851e+07 | - |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | - | 200.794500 | - |
| classical2 | gmm | istella | n_iter | - | 24 | - |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.668e+06 | - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.807640 | - |
| classical2 | gmm | taxi | n_iter | - | 29 | - |
| classical2 | ivf | istella | recall_at_k (higher is better) | - | 1.000000 | faiss-cpu - |
| classical2 | ivf | istella | rows_with_repeated_ids | - | 0 | faiss-cpu - |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | 0.999675 | faiss-cpu - |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | 0 | faiss-cpu - |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328682 | sklearn-cpu 0.328676 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684423 | sklearn-cpu 0.684426 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | sklearn-cpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805050 | sklearn-cpu 4.805056 |
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
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.980129 | catboost-cpu 0.979899 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.186686 | catboost-cpu 0.188093 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.630436 | catboost-cpu 0.630269 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.528554 | catboost-cpu 0.528650 |

## Trees

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10258.4 | 10258.4..10258.4 | 1 | - | - | 10258.4 | - | - (stored whole) | - | - | - | - | auc=0.631297, logloss=0.528285 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | 97857.4 | 97857.4..97857.4 | 1 | 0.105 | - | 97857.4 | 97857.4 | 0.00 (cpu-arm) | 0.105 (whole/whole) | - | 11918.4 | - | auc=0.628772, logloss=0.528923 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 18924.8 | 18924.8..18924.8 | 1 | 0.542 | - | 18924.8 | 18924.8 | 0.00 (cpu-arm) | 0.542 (whole/whole) | - | 8838.5 | - | auc=0.631473, logloss=0.528686 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 10928.0 | 10928.0..10928.0 | 1 | 0.939 | - | 10928.0 | 10928.0 | 0.00 (cpu-arm) | 0.939 (whole/whole) | - | 8741.9 | - | auc=0.632665, logloss=0.528094 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6295.4 | 6295.4..6295.4 | 1 | - | - | 6295.4 | - | - (stored whole) | - | - | - | - | auc=0.983304, logloss=0.156072 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | 70785.7 | 70785.7..70785.7 | 1 | 0.089 | - | 70785.7 | 70785.7 | 0.00 (cpu-arm) | 0.089 (whole/whole) | - | 10524.7 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 16034.8 | 16034.8..16034.8 | 1 | 0.393 | - | 16034.8 | 16034.8 | 0.00 (cpu-arm) | 0.393 (whole/whole) | - | 11639.5 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3631.3 | 3631.3..3631.3 | 1 | - | - | 3631.3 | - | - (stored whole) | - | - | - | - | auc=0.632205, logloss=0.527838 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | 43791.1 | 43791.1..43791.1 | 1 | 0.083 | - | 43791.1 | 43791.1 | 0.00 (cpu-arm) | 0.083 (whole/whole) | - | 7279.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 8079.8 | 8079.8..8079.8 | 1 | 0.449 | - | 8079.8 | 8079.8 | 0.00 (cpu-arm) | 0.449 (whole/whole) | - | 7413.2 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6462.9 | 6462.9..6462.9 | 1 | - | - | 6462.9 | - | - (stored whole) | - | - | - | - | auc=0.983586, logloss=0.150066 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | 103179.7 | 103179.7..103179.7 | 1 | 0.063 | - | 103179.7 | 103179.7 | 0.00 (cpu-arm) | 0.063 (whole/whole) | - | 10528.0 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 23613.1 | 23613.1..23613.1 | 1 | 0.274 | - | 23613.1 | 23613.1 | 0.00 (cpu-arm) | 0.274 (whole/whole) | - | 11234.1 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 19843.3 | 19843.3..19843.3 | 1 | 0.326 | - | 19843.3 | 19843.3 | 0.00 (cpu-arm) | 0.326 (whole/whole) | - | 11308.2 | - | auc=0.983778, logloss=0.149653 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3662.6 | 3662.6..3662.6 | 1 | - | - | 3662.6 | - | - (stored whole) | - | - | - | - | auc=0.632096, logloss=0.528044 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | 77432.0 | 77432.0..77432.0 | 1 | 0.047 | - | 77432.0 | 77432.0 | 0.00 (cpu-arm) | 0.047 (whole/whole) | - | 6674.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 10350.4 | 10350.4..10350.4 | 1 | 0.354 | - | 10350.4 | 10350.4 | 0.00 (cpu-arm) | 0.354 (whole/whole) | - | 6853.6 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 10099.2 | 10099.2..10099.2 | 1 | 0.363 | - | 10099.2 | 10099.2 | 0.00 (cpu-arm) | 0.363 (whole/whole) | - | 6933.8 | - | auc=0.632243, logloss=0.528067 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10032.0 | 10032.0..10032.0 | 1 | - | - | 10032.0 | - | - (stored whole) | - | - | - | - | accuracy=0.907556, mlogloss=0.258413 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | 561543.4 | 561543.4..561543.4 | 1 | 0.018 | - | 561543.4 | 561543.4 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 11547.8 | - | accuracy=0.907768, mlogloss=0.258286 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 76053.9 | 76053.9..76053.9 | 1 | 0.132 | - | 76053.9 | 76053.9 | 0.00 (cpu-arm) | 0.132 (whole/whole) | - | 12606.4 | - | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 79078.6 | 79078.6..79078.6 | 1 | 0.127 | - | 79078.6 | 79078.6 | 0.00 (cpu-arm) | 0.127 (whole/whole) | - | 12549.6 | - | accuracy=0.910058, mlogloss=0.245916 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6049.1 | 6049.1..6049.1 | 1 | - | - | 6049.1 | - | - (stored whole) | - | - | - | - | accuracy=0.599380, mlogloss=1.012595 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | - | - | - | - | UNKNOWN | - | REFUSED(asked for by --arms and not built here; built: xgboost-cpu) (copied from opponents-20261006; measured this run) |
| catboost-cpu | catboost | cpu | opponent | 178349.9 | 178349.9..178349.9 | 1 | 0.034 | - | 178349.9 | 178349.9 | 0.00 (cpu-arm) | 0.034 (whole/whole) | - | 8021.3 | - | accuracy=0.599150, mlogloss=1.012734 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 35861.4 | 35861.4..35861.4 | 1 | 0.169 | - | 35861.4 | 35861.4 | 0.00 (cpu-arm) | 0.169 (whole/whole) | - | 8463.4 | - | accuracy=0.601128, mlogloss=1.005128 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 41539.9 | 41539.9..41539.9 | 1 | 0.146 | - | 41539.9 | 41539.9 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 8711.3 | - | accuracy=0.601580, mlogloss=1.004282 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours, xgboost-gpu: host not sampled; GPU not sampled

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 36196.7 | 36196.7..36196.7 | 1 | - | - | 36196.7 | - | - (stored whole) | - | - | - | - | auc=0.979444, logloss=0.190928 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| catboost-cpu | catboost | cpu | opponent | 273122.4 | 273122.4..273122.4 | 1 | 0.133 | - | 273122.4 | 273122.4 | 0.00 (cpu-arm) | 0.133 (whole/whole) | - | 11159.1 | - | auc=0.979221, logloss=0.192114 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22025.3 | 22025.3..22025.3 | 1 | - | - | 22025.3 | - | - (stored whole) | - | - | - | - | auc=0.629203, logloss=0.529007 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| catboost-cpu | catboost | cpu | opponent | 78566.1 | 78566.1..78566.1 | 1 | 0.280 | - | 78566.1 | 78566.1 | 0.00 (cpu-arm) | 0.280 (whole/whole) | - | 8753.0 | - | auc=0.628918, logloss=0.529083 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5995.7 | 5995.7..5995.7 | 1 | - | - | 5995.7 | - | - (stored whole) | - | - | - | - | auc=0.980129, logloss=0.186686 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| catboost-cpu | catboost | cpu | opponent | 44398.5 | 44398.5..44398.5 | 1 | 0.135 | - | 44398.5 | 44398.5 | 0.00 (cpu-arm) | 0.135 (whole/whole) | - | 9400.6 | - | auc=0.979899, logloss=0.188093 | yes | NOT-COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0499)`, ran on amd (DigitalOcean MI325X) job a0499

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3958.8 | 3958.8..3958.8 | 1 | - | - | 3958.8 | - | - (stored whole) | - | - | - | - | auc=0.630436, logloss=0.528554 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0499 2026-10-08; identity vs nvidia-l40s: n/a) |
| catboost-cpu | catboost | cpu | opponent | 21286.9 | 21286.9..21286.9 | 1 | 0.186 | - | 21286.9 | 21286.9 | 0.00 (cpu-arm) | 0.186 (whole/whole) | - | 5984.4 | - | auc=0.630269, logloss=0.528650 | yes | COMPARABLE | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on amd (DigitalOcean MI325X) job a0536

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 412.9 | 412.9..412.9 | 1 | - | - | 412.9 | - | - (stored whole) | - | - | - | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 3406.3 | 3406.3..3406.3 | 1 | 0.121 | - | 3592.3 | 3406.3 | 186.04 (upload_ms_untimed) | 0.115 (whole/whole (kernel not derivable)) | - | 5132.4 | 3460.8 | inertia=5.991e+17, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 5614.6 | 5614.6..5614.6 | 1 | 0.074 | - | 5614.6 | 5614.6 | 0.00 (cpu-arm) | 0.074 (whole/whole) | - | 5794.7 | - | inertia=6.049e+17, inertia_over_ours=0.999759, n_iter=24 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:34:26Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on amd (DigitalOcean MI325X) job a0536

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 209.8 | 209.8..209.8 | 1 | - | - | 209.8 | - | - (stored whole) | - | - | - | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 538.0 | 538.0..538.0 | 1 | 0.390 | - | 684.4 | 538.0 | 146.41 (upload_ms_untimed) | 0.306 (whole/whole (kernel not derivable)) | - | 3194.4 | 459.9 | inertia=3.06e+08, n_iter=54 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2297.6 | 2297.6..2297.6 | 1 | 0.091 | - | 2297.6 | 2297.6 | 0.00 (cpu-arm) | 0.091 (whole/whole) | - | 772.1 | - | inertia=3.093e+08, inertia_over_ours=0.999937, n_iter=58 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:33:52Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0565)`, ran on amd (DigitalOcean MI325X) job a0565

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 495.2 | 495.2..495.2 | 1 | - | - | 495.2 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.332506, rmse=0.681740 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0565 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 2340.8 | 2340.8..2340.8 | 1 | 0.212 | - | 2523.3 | 2340.8 | 182.42 (upload_ms_untimed) | 0.196 (whole/whole (kernel not derivable)) | - | 5648.2 | 5303.6 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 42.8 | 42.8..42.8 | 1 | 11.568 | - | 222.4 | 42.8 | 179.55 (upload_ms_untimed) | 2.227 (whole/whole (kernel not derivable)) | - | 5051.5 | 3649.4 | finite=True, r2=0.151604, rmse=0.768589 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3980.6 | 3980.6..3980.6 | 1 | 0.124 | - | 3980.6 | 3980.6 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 5793.0 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:47Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

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

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0565)`, ran on amd (DigitalOcean MI325X) job a0565

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.9 | 7.9..7.9 | 1 | - | - | 7.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908836, rmse=4.696479 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0565 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 100.7 | 100.7..100.7 | 1 | 0.079 | - | 243.7 | 100.7 | 143.01 (upload_ms_untimed) | 0.032 (whole/whole (kernel not derivable)) | - | 3535.9 | 695.3 | finite=True, r2=0.908840, rmse=4.696376 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 34.0 | 34.0..34.0 | 1 | 0.233 | - | 176.6 | 34.0 | 142.61 (upload_ms_untimed) | 0.045 (whole/whole (kernel not derivable)) | - | 3018.3 | 572.0 | finite=True, r2=0.908822, rmse=4.696849 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 304.7 | 304.7..304.7 | 1 | 0.026 | - | 304.7 | 304.7 | 0.00 (cpu-arm) | 0.026 (whole/whole) | - | 782.5 | - | finite=True, r2=0.724850, rmse=8.159187 | yes | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; stored (measured 2026-09-29T16:35:14Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Classical, wave 2

### gmm / istella (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on amd (DigitalOcean MI325X) job a0536

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 541.5 | 541.5..541.5 | 1 | - | - | 541.5 | - | - (stored whole) | - | - | - | - | bic=-3.851e+07, mean_log_likelihood=200.794500, n_iter=24 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gmm / taxi (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on amd (DigitalOcean MI325X) job a0536

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56.3 | 56.3..56.3 | 1 | - | - | 56.3 | - | - (stored whole) | - | - | - | - | bic=-3.668e+06, mean_log_likelihood=12.807640, n_iter=29 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows); an explicitly full_dataset_coverage recipe retains all fit/eval rows. Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

mismatch: opponents withheld: the lane settings at this tree's HEAD differ from the settings release-board-resume-r2 recorded for its opponent race (an opponent job must score them again)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0881)`, ran on amd (DigitalOcean MI325X) job a0881

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1144.0 | 1144.0..1144.0 | 1 | - | - | 1144.0 | - | - (stored whole) | - | - | - | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0881 2026-10-08; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | opponent | 3704.2 | 3704.2..3704.2 | 1 | 0.309 | - | 3704.2 | 3704.2 | 0.00 (cpu-arm) | 0.309 (whole/whole) | - | 1286.6 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0881)`, ran on amd (DigitalOcean MI325X) job a0881

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 168.6 | 168.6..168.6 | 1 | - | - | 168.6 | - | - (stored whole) | - | - | - | - | recall_at_k=0.999675, rows_with_repeated_ids=0 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0881 2026-10-08; identity vs nvidia-l40s: MATCH) |
| faiss-cpu | faiss | cpu | opponent | 143.0 | 143.0..143.0 | 1 | 1.180 | - | 143.0 | 143.0 | 0.00 (cpu-arm) | 1.180 (whole/whole) | - | 139.0 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on amd (DigitalOcean MI325X) job a0536

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1075.7 | 1075.7..1075.7 | 1 | - | - | 1075.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328682, rmse=0.684423 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 4288.8 | 4288.8..4288.8 | 1 | 0.251 | - | 4288.8 | 4288.8 | 0.00 (cpu-arm) | 0.251 (whole/whole) | - | 8688.9 | - | finite=True, r2=0.328676, rmse=0.684426 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-grid-logs.txt (amd/a0536)`, ran on amd (DigitalOcean MI325X) job a0536

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.1 | 3.1..3.1 | 1 | - | - | 3.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908983, rmse=4.805050 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0536 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 33.2 | 33.2..33.2 | 1 | 0.093 | - | 33.2 | 33.2 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 291.3 | - | finite=True, r2=0.908983, rmse=4.805056 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on amd (DigitalOcean MI325X) job a0873

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.8 | 8.8..8.8 | 1 | - | - | 8.8 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-adafactor-synthetic-def/adafactor-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 7.5 | 7.5..7.5 | 1 | 1.171 | - | - | 7.5 | - (stored kernel) | 1.171 (MIXED ours whole / arm kernel) | - | 2842.8 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 65.3 | 65.3..65.3 | 1 | 0.135 | - | - | 65.3 | - (stored kernel) | 0.135 (MIXED ours whole / arm kernel) | - | 2887.5 | 960.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adagrad / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on amd (DigitalOcean MI325X) job a0873

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.5 | 3.5..3.5 | 1 | - | - | 3.5 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-adagrad-synthetic-def/adagrad-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 4.7 | 4.7..4.7 | 1 | 0.740 | - | - | 4.7 | - (stored kernel) | 0.740 (MIXED ours whole / arm kernel) | - | 2831.4 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 282.7 | 282.7..282.7 | 1 | 0.012 | - | - | 282.7 | - (stored kernel) | 0.012 (MIXED ours whole / arm kernel) | - | 2923.9 | 832.0 | rel_fro_vs_torch_eager_fp32=1.603e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### adamax / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on amd (DigitalOcean MI325X) job a0873

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.8 | 3.8..3.8 | 1 | - | - | 3.8 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-adamax-synthetic-def/adamax-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 5.7 | 5.7..5.7 | 1 | 0.664 | - | - | 5.7 | - (stored kernel) | 0.664 (MIXED ours whole / arm kernel) | - | 2836.1 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 299.2 | 299.2..299.2 | 1 | 0.013 | - | - | 299.2 | - (stored kernel) | 0.013 (MIXED ours whole / arm kernel) | - | 2925.6 | 896.0 | rel_fro_vs_torch_eager_fp32=6.17e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on amd (DigitalOcean MI325X) job a0866

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 163.4 | 163.4..163.4 | 1 | - | - | 163.4 | - | - (stored whole) | - | - | - | - | relative_residual=2.9e-07 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs nvidia-l40s: n/a) |
| torch-gpu | torch | gpu | opponent | 20.1 | 20.1..20.1 | 1 | 8.146 | - | - | 20.1 | - (stored kernel) | 8.146 (MIXED ours whole / arm kernel) | - | 3745.3 | 768.0 | relative_residual=1.044e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 7138.9 | 7138.9..7138.9 | 1 | 0.023 | - | 7138.9 | 7138.9 | 0.00 (cpu-arm) | 0.023 (whole/whole) | - | 2650.2 | - | relative_residual=3.928e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0879)`, ran on amd (DigitalOcean MI325X) job a0879

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21.0 | 21.0..21.0 | 1 | - | - | 21.0 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559491 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0879 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 269.7 | 269.7..269.7 | 1 | 0.078 | - | 269.7 | 269.7 | 0.00 (cpu-arm) | 0.078 (whole/whole) | - | 4455.8 | - | accuracy=0.983067, logloss=0.557285 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on amd (DigitalOcean MI325X) job a0870

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.7 | 0.7..0.7 | 1 | - | - | 0.7 | - | - (stored whole) | - | - | - | - | n_components=81 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | opponent | 5.7 | 5.7..5.7 | 1 | 0.118 | - | 5.7 | 5.7 | 0.00 (cpu-arm) | 0.118 (whole/whole) | - | 117.4 | - | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on amd (DigitalOcean MI325X) job a0870

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.6 | 0.6..0.6 | 1 | - | - | 0.6 | - | - (stored whole) | - | - | - | - | n_components=588 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | opponent | 5.3 | 5.3..5.3 | 1 | 0.122 | - | 5.3 | 5.3 | 0.00 (cpu-arm) | 0.122 (whole/whole) | - | 99.9 | - | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on amd (DigitalOcean MI325X) job a1069

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1679.7 | 1679.7..1679.7 | 1 | - | - | 1679.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=13.931153 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs nvidia-l40s: n/a) |
| statsmodels-cpu | statsmodels | cpu | opponent | 94.5 | 94.5..94.5 | 1 | 17.782 | - | 94.5 | 94.5 | 0.00 (cpu-arm) | 17.782 (whole/whole) | - | 58.1 | - | forecast_rmse=26.588738 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | opponent | 93.0 | 93.0..93.0 | 1 | 18.056 | - | 93.0 | 93.0 | 0.00 (cpu-arm) | 18.056 (whole/whole) | - | 283.5 | - | forecast_rmse=13.945986 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### damped-ets / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on amd (DigitalOcean MI325X) job a1069

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1770.6 | 1770.6..1770.6 | 1 | - | - | 1770.6 | - | - (stored whole) | - | - | - | - | forecast_rmse=96.690449 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs nvidia-l40s: n/a) |
| statsmodels-cpu | statsmodels | cpu | opponent | 94.0 | 94.0..94.0 | 1 | 18.828 | - | 94.0 | 94.0 | 0.00 (cpu-arm) | 18.828 (whole/whole) | - | 57.9 | - | forecast_rmse=196.955273 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsforecast-cpu | statsforecast | cpu | opponent | 63.3 | 63.3..63.3 | 1 | 27.968 | - | 63.3 | 63.3 | 0.00 (cpu-arm) | 27.968 (whole/whole) | - | 283.6 | - | forecast_rmse=96.685568 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsmodels-cpu, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'damped': True, 'model': 'AAN', 'season_length': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### eigh / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on amd (DigitalOcean MI325X) job a0866

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8620.1 | 8620.1..8620.1 | 1 | - | - | 8620.1 | - | - (stored whole) | - | - | - | - | max_eigenvalue_error=5.95e-05, relative_residual=5.251e-05 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs nvidia-l40s: n/a) |
| torch-gpu | torch | gpu | opponent | 1922.1 | 1922.1..1922.1 | 1 | 4.485 | - | - | 1922.1 | - (stored kernel) | 4.485 (MIXED ours whole / arm kernel) | - | 2900.1 | 1220.6 | max_eigenvalue_error=1.254e-06, relative_residual=1.269e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 2356.5 | 2356.5..2356.5 | 1 | 3.658 | - | 2356.5 | 2356.5 | 0.00 (cpu-arm) | 3.658 (whole/whole) | - | 970.0 | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on amd (DigitalOcean MI325X) job a1065

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 837.3 | 837.3..837.3 | 1 | - | - | 837.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.316583, rmse=0.690563 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 47198.6 | 47198.6..47198.6 | 1 | 0.018 | - | 47198.6 | 47198.6 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 8285.3 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on amd (DigitalOcean MI325X) job a1065

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 328.6 | 328.6..328.6 | 1 | - | - | 328.6 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909002, rmse=4.804540 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 279.5 | 279.5..279.5 | 1 | 1.176 | - | 279.5 | 279.5 | 0.00 (cpu-arm) | 1.176 (whole/whole) | - | 673.5 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on amd (DigitalOcean MI325X) job a1066

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 46.4 | 46.4..46.4 | 1 | - | - | 46.4 | - | - (stored whole) | - | - | - | - | accuracy=0.876570, logloss=3.574420 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 420.3 | 420.3..420.3 | 1 | 0.110 | - | 420.3 | 420.3 | 0.00 (cpu-arm) | 0.110 (whole/whole) | - | 2633.4 | - | accuracy=0.876530, logloss=3.417392 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on amd (DigitalOcean MI325X) job a1066

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.1 | 14.1..14.1 | 1 | - | - | 14.1 | - | - (stored whole) | - | - | - | - | accuracy=0.719820, logloss=1.132249 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 82.1 | 82.1..82.1 | 1 | 0.172 | - | 82.1 | 82.1 | 0.00 (cpu-arm) | 0.172 (whole/whole) | - | 324.2 | - | accuracy=0.719900, logloss=1.133898 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1067)`, ran on amd (DigitalOcean MI325X) job a1067

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 143778.0 | 143778.0..143778.0 | 1 | - | - | 143778.0 | - | - (stored whole) | - | - | - | - | recall_at_10=0.967575 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1067 2026-10-09; identity vs nvidia-l40s: n/a) |
| faiss-cpu | faiss | cpu | opponent | 914.0 | 914.0..914.0 | 1 | 157.304 | - | 914.0 | 914.0 | 0.00 (cpu-arm) | 157.304 (whole/whole) | - | 142.7 | - | recall_at_10=0.979450 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on amd (DigitalOcean MI325X) job a0904

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8230.1 | 8230.1..8230.1 | 1 | - | - | 8230.1 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=8.307e-08 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | opponent | 10339.5 | 10339.5..10339.5 | 1 | 0.796 | - | 10339.5 | 10339.5 | 0.00 (cpu-arm) | 0.796 (whole/whole) | - | 2196.3 | - | rel_error_vs_exact=8.63e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### kernel-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on amd (DigitalOcean MI325X) job a0904

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 252.6 | 252.6..252.6 | 1 | - | - | 252.6 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.056e-07 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | opponent | 1154.2 | 1154.2..1154.2 | 1 | 0.219 | - | 1154.2 | 1154.2 | 0.00 (cpu-arm) | 0.219 (whole/whole) | - | 405.9 | - | rel_error_vs_exact=8.149e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / istella (rows full, shape X 100000x220; X_true 100000x220; Xq 20000x220; Xq_true 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0871)`, ran on amd (DigitalOcean MI325X) job a0871

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24.0 | 24.0..24.0 | 1 | - | - | 24.0 | - | - (stored whole) | - | - | - | - | masked_rmse=323953.237332 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0871 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 16.2 | 16.2..16.2 | 1 | 1.477 | - | 16.2 | 16.2 | 0.00 (cpu-arm) | 1.477 (whole/whole) | - | 3528.4 | - | masked_rmse=986208.700423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0871)`, ran on amd (DigitalOcean MI325X) job a0871

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.1 | 1.1..1.1 | 1 | - | - | 1.1 | - | - (stored whole) | - | - | - | - | masked_rmse=6.151696 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0871 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.9 | 1.9..1.9 | 1 | 0.555 | - | 1.9 | 1.9 | 0.00 (cpu-arm) | 0.555 (whole/whole) | - | 1966.8 | - | masked_rmse=5.263919 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lamb / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on amd (DigitalOcean MI325X) job a0873

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.4 | 13.4..13.4 | 1 | - | - | 13.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-lamb-synthetic-def/lamb-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'betas': [0.9, 0.999], 'eps': 1e-06, 'lr': 0.001, 'weight_decay': 0.01}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on amd (DigitalOcean MI325X) job a1065

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 853.7 | 853.7..853.7 | 1 | - | - | 853.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.310329, rmse=0.693715 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 42194.4 | 42194.4..42194.4 | 1 | 0.020 | - | 42194.4 | 42194.4 | 0.00 (cpu-arm) | 0.020 (whole/whole) | - | 8912.8 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on amd (DigitalOcean MI325X) job a1065

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 414.8 | 414.8..414.8 | 1 | - | - | 414.8 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.909059, rmse=4.803051 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 277.2 | 277.2..277.2 | 1 | 1.496 | - | 277.2 | 277.2 | 0.00 (cpu-arm) | 1.496 (whole/whole) | - | 680.5 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on amd (DigitalOcean MI325X) job a1066

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 315.8 | 315.8..315.8 | 1 | - | - | 315.8 | - | - (stored whole) | - | - | - | - | accuracy=0.912830, logloss=0.235875 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 3925.4 | 3925.4..3925.4 | 1 | 0.080 | - | 3925.4 | 3925.4 | 0.00 (cpu-arm) | 0.080 (whole/whole) | - | 5444.6 | - | accuracy=0.899520, logloss=0.439049 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lda-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on amd (DigitalOcean MI325X) job a1066

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17.4 | 17.4..17.4 | 1 | - | - | 17.4 | - | - (stored whole) | - | - | - | - | accuracy=0.762530, logloss=0.539763 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 230.9 | 230.9..230.9 | 1 | 0.075 | - | 230.9 | 230.9 | 0.00 (cpu-arm) | 0.075 (whole/whole) | - | 513.8 | - | accuracy=0.762530, logloss=0.539767 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'solver': 'svd', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lion / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on amd (DigitalOcean MI325X) job a0873

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-lion-synthetic-def/lion-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |

settings: {'betas': [0.9, 0.99], 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on amd (DigitalOcean MI325X) job a0870

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 359.9 | 359.9..359.9 | 1 | - | - | 359.9 | - | - (stored whole) | - | - | - | - | modularity=0.911187, n_communities=40 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | opponent | 1756.7 | 1756.7..1756.7 | 1 | 0.205 | - | 1756.7 | 1756.7 | 0.00 (cpu-arm) | 0.205 (whole/whole) | - | 238.5 | - | modularity=0.908460, n_communities=40 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### louvain / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on amd (DigitalOcean MI325X) job a0870

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 71.2 | 71.2..71.2 | 1 | - | - | 71.2 | - | - (stored whole) | - | - | - | - | modularity=0.941953, n_communities=58 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | opponent | 1078.3 | 1078.3..1078.3 | 1 | 0.066 | - | 1078.3 | 1078.3 | 0.00 (cpu-arm) | 0.066 (whole/whole) | - | 203.7 | - | modularity=0.940781, n_communities=56 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on amd (DigitalOcean MI325X) job a0868

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 288.9 | 288.9..288.9 | 1 | - | - | 288.9 | - | - (stored whole) | - | - | - | - | relative_residual=0.849956 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 1138.9 | 1138.9..1138.9 | 1 | 0.254 | - | - | 1138.9 | - (stored kernel) | 0.254 (MIXED ours whole / arm kernel) | - | 4308.1 | 1691.4 | relative_residual=nan | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 2526.8 | 2526.8..2526.8 | 1 | 0.114 | - | 2526.8 | 2526.8 | 0.00 (cpu-arm) | 0.114 (whole/whole) | - | 4351.2 | - | relative_residual=0.876106 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on amd (DigitalOcean MI325X) job a0868

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.7 | 7.7..7.7 | 1 | - | - | 7.7 | - | - (stored whole) | - | - | - | - | relative_residual=0.756366 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 33.3 | 33.3..33.3 | 1 | 0.231 | - | - | 33.3 | - (stored kernel) | 0.231 (MIXED ours whole / arm kernel) | - | 3132.5 | 95.4 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 91.5 | 91.5..91.5 | 1 | 0.084 | - | 91.5 | 91.5 | 0.00 (cpu-arm) | 0.084 (whole/whole) | - | 284.0 | - | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on amd (DigitalOcean MI325X) job a0866

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 574.6 | 574.6..574.6 | 1 | - | - | 574.6 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs nvidia-l40s: n/a) |
| torch-gpu | torch | gpu | opponent | 86.9 | 86.9..86.9 | 1 | 6.613 | - | - | 86.9 | - (stored kernel) | 6.613 (MIXED ours whole / arm kernel) | - | 2673.1 | 710.0 | relative_residual=4.003e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| scipy-cpu | scipy | cpu | opponent | 4553.9 | 4553.9..4553.9 | 1 | 0.126 | - | 4553.9 | 4553.9 | 0.00 (cpu-arm) | 0.126 (whole/whole) | - | 628.0 | - | relative_residual=3.439e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, scipy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0866)`, ran on amd (DigitalOcean MI325X) job a0866

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 324.2 | 324.2..324.2 | 1 | - | - | 324.2 | - | - (stored whole) | - | - | - | - | relative_residual=3.256e-06 | - | main board, one scored run | - | ok (main@bb7cc2fa0 amd/a0866 2026-10-08; identity vs nvidia-l40s: n/a) |
| torch-gpu | torch | gpu | opponent | 86.3 | 86.3..86.3 | 1 | 3.755 | - | - | 86.3 | - (stored kernel) | 3.755 (MIXED ours whole / arm kernel) | - | 2694.0 | 712.0 | relative_residual=4.041e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 5413.8 | 5413.8..5413.8 | 1 | 0.060 | - | 5413.8 | 5413.8 | 0.00 (cpu-arm) | 0.060 (whole/whole) | - | 1389.2 | - | relative_residual=3.259e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0879)`, ran on amd (DigitalOcean MI325X) job a0879

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17.7 | 17.7..17.7 | 1 | - | - | 17.7 | - | - (stored whole) | - | - | - | - | accuracy=0.983067, logloss=0.559529 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0879 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 271.3 | 271.3..271.3 | 1 | 0.065 | - | 271.3 | 271.3 | 0.00 (cpu-arm) | 0.065 (whole/whole) | - | 4455.8 | - | accuracy=0.983067, logloss=0.557319 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### nadam / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on amd (DigitalOcean MI325X) job a0873

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.1 | 4.1..4.1 | 1 | - | - | 4.1 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-nadam-synthetic-def/nadam-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 7.0 | 7.0..7.0 | 1 | 0.585 | - | - | 7.0 | - (stored kernel) | 0.585 (MIXED ours whole / arm kernel) | - | 2835.0 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 333.5 | 333.5..333.5 | 1 | 0.012 | - | - | 333.5 | - (stored kernel) | 0.012 (MIXED ours whole / arm kernel) | - | 2927.7 | 896.0 | rel_fro_vs_torch_eager_fp32=1.464e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22.5 | 22.5..22.5 | 1 | - | - | 22.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x119 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 13.3 | 13.3..13.3 | 1 | 1.701 | - | 13.3 | 13.3 | 0.00 (cpu-arm) | 1.701 (whole/whole) | - | 539.3 | - | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19.1 | 19.1..19.1 | 1 | - | - | 19.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x508 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 8.0 | 8.0..8.0 | 1 | 2.374 | - | 8.0 | 8.0 | 0.00 (cpu-arm) | 2.374 (whole/whole) | - | 1407.9 | - | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on amd (DigitalOcean MI325X) job a1069

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 626.5 | 626.5..626.5 | 1 | - | - | 626.5 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.438855 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs nvidia-l40s: n/a) |
| statsforecast-cpu | statsforecast | cpu | opponent | 303.9 | 303.9..303.9 | 1 | 2.061 | - | 303.9 | 303.9 | 0.00 (cpu-arm) | 2.061 (whole/whole) | - | 283.6 | - | forecast_rmse=1.437815 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### optimized-theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on amd (DigitalOcean MI325X) job a1069

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1116.7 | 1116.7..1116.7 | 1 | - | - | 1116.7 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.150860 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs nvidia-l40s: n/a) |
| statsforecast-cpu | statsforecast | cpu | opponent | 770.4 | 770.4..770.4 | 1 | 1.450 | - | 770.4 | 770.4 | 0.00 (cpu-arm) | 1.450 (whole/whole) | - | 283.5 | - | forecast_rmse=49.356608 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22.1 | 22.1..22.1 | 1 | - | - | 22.1 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 13.6 | 13.6..13.6 | 1 | 1.631 | - | 13.6 | 13.6 | 0.00 (cpu-arm) | 1.631 (whole/whole) | - | 267.7 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ordinal / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 18.5 | 18.5..18.5 | 1 | - | - | 18.5 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 7.4 | 7.4..7.4 | 1 | 2.484 | - | 7.4 | 7.4 | 0.00 (cpu-arm) | 2.484 (whole/whole) | - | 246.0 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OrdinalEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on amd (DigitalOcean MI325X) job a0870

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.4 | 2.4..2.4 | 1 | - | - | 2.4 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | opponent | 115.0 | 115.0..115.0 | 1 | 0.021 | - | 115.0 | 115.0 | 0.00 (cpu-arm) | 0.021 (whole/whole) | - | 213.2 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0870)`, ran on amd (DigitalOcean MI325X) job a0870

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.3 | 2.3..2.3 | 1 | - | - | 2.3 | - | - (stored whole) | - | - | - | - | sum=1.000000 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0870 2026-10-08; identity vs nvidia-l40s: MATCH) |
| networkx-cpu | networkx | cpu | opponent | 88.5 | 88.5..88.5 | 1 | 0.026 | - | 88.5 | 88.5 | 0.00 (cpu-arm) | 0.026 (whole/whole) | - | 178.4 | - | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, networkx-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on amd (DigitalOcean MI325X) job a0904

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1979.0 | 1979.0..1979.0 | 1 | - | - | 1979.0 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=6.68e-08 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | opponent | 22453.9 | 22453.9..22453.9 | 1 | 0.088 | - | 22453.9 | 22453.9 | 0.00 (cpu-arm) | 0.088 (whole/whole) | - | 1842.9 | - | rel_error_vs_exact=3.692e-10 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0904)`, ran on amd (DigitalOcean MI325X) job a0904

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 31.8 | 31.8..31.8 | 1 | - | - | 31.8 | - | - (stored whole) | - | - | - | - | rel_error_vs_exact=1.054e-07 | - | main board, one scored run | - | ok (main@03b648834 amd/a0904 2026-10-09; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | opponent | 104.1 | 104.1..104.1 | 1 | 0.305 | - | 104.1 | 104.1 | 0.00 (cpu-arm) | 0.305 (whole/whole) | - | 478.9 | - | rel_error_vs_exact=1.218e-15 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on amd (DigitalOcean MI325X) job a0868

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 724.8 | 724.8..724.8 | 1 | - | - | 724.8 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.485e-07 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 1852.3 | 1852.3..1852.3 | 1 | 0.391 | - | - | 1852.3 | - (stored kernel) | 0.391 (MIXED ours whole / arm kernel) | - | 3625.8 | 2520.6 | relative_gram_difference=0.0001698 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 4646.6 | 4646.6..4646.6 | 1 | 0.156 | - | 4646.6 | 4646.6 | 0.00 (cpu-arm) | 0.156 (whole/whole) | - | 8532.1 | - | relative_gram_difference=2.462e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on amd (DigitalOcean MI325X) job a0868

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16.5 | 16.5..16.5 | 1 | - | - | 16.5 | - | - (stored whole) | - | - | - | - | relative_gram_difference=1.714e-07 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 24.6 | 24.6..24.6 | 1 | 0.669 | - | - | 24.6 | - (stored kernel) | 0.669 (MIXED ours whole / arm kernel) | - | 2690.6 | 126.0 | relative_gram_difference=6.794e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 177.3 | 177.3..177.3 | 1 | 0.093 | - | 177.3 | 177.3 | 0.00 (cpu-arm) | 0.093 (whole/whole) | - | 478.4 | - | relative_gram_difference=3.024e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0535)`, ran on amd (DigitalOcean MI325X) job a0535

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 862.1 | 862.1..862.1 | 1 | - | - | 862.1 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.039998, rmse=0.851877 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0535 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 878876.6 | 878876.6..878876.6 | 1 | 0.0009809 | - | 878876.6 | 878876.6 | 0.00 (cpu-arm) | 0.0009809 (whole/whole) | - | 8006.1 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0535)`, ran on amd (DigitalOcean MI325X) job a0535

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 499.3 | 499.3..499.3 | 1 | - | - | 499.3 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.899596, rmse=5.046749 | - | main board, one scored run | - | ok (main@de2b2b739 amd/a0535 2026-10-08; identity vs nvidia-l40s: MATCH) |
| sklearn-cpu | scikit-learn | cpu | opponent | 225040.2 | 225040.2..225040.2 | 1 | 0.002 | - | 225040.2 | 225040.2 | 0.00 (cpu-arm) | 0.002 (whole/whole) | - | 928.9 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on amd (DigitalOcean MI325X) job a1066

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 212.7 | 212.7..212.7 | 1 | - | - | 212.7 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.0002359 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs nvidia-l40s: n/a) |
| torch-gpu | torch | gpu | opponent | 352.0 | 352.0..352.0 | 1 | 0.604 | - | - | 352.0 | - (stored kernel) | 0.604 (MIXED ours whole / arm kernel) | - | 3836.8 | 1017.6 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 812.6 | 812.6..812.6 | 1 | 0.262 | - | 812.6 | 812.6 | 0.00 (cpu-arm) | 0.262 (whole/whole) | - | 1710.8 | - | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1066)`, ran on amd (DigitalOcean MI325X) job a1066

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 98.2 | 98.2..98.2 | 1 | - | - | 98.2 | - | - (stored whole) | - | - | - | - | relative_reconstruction_error=0.027197 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1066 2026-10-09; identity vs nvidia-l40s: n/a) |
| torch-gpu | torch | gpu | opponent | 215.0 | 215.0..215.0 | 1 | 0.457 | - | - | 215.0 | - (stored kernel) | 0.457 (MIXED ours whole / arm kernel) | - | 3037.1 | 228.0 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 375.4 | 375.4..375.4 | 1 | 0.262 | - | 375.4 | 375.4 | 0.00 (cpu-arm) | 0.262 (whole/whole) | - | 476.9 | - | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 62958.7 | 62958.7..62958.7 | 1 | - | - | 62958.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.328684, rmse=0.684422 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 78354.8 | 78354.8..78354.8 | 1 | 0.804 | - | 78354.8 | 78354.8 | 0.00 (cpu-arm) | 0.804 (whole/whole) | - | 8705.1 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 288.9 | 288.9..288.9 | 1 | - | - | 288.9 | - | - (stored whole) | - | - | - | - | finite=True, r2=0.908981, rmse=4.805109 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 994.6 | 994.6..994.6 | 1 | 0.290 | - | 994.6 | 994.6 | 0.00 (cpu-arm) | 0.290 (whole/whole) | - | 380.7 | - | finite=True, r2=0.908983, rmse=4.805057 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0873)`, ran on amd (DigitalOcean MI325X) job a0873

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.4 | 3.4..3.4 | 1 | - | - | 3.4 | - | - (stored whole) | - | - | - | - | error=Own host reference failed; see /root/lq/out/a0873/work-rmsprop-synthetic-def/rmsprop-synthetic-host-quality/host.log | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0873 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-eager-fp32 | torch | gpu | opponent | 5.1 | 5.1..5.1 | 1 | 0.664 | - | - | 5.1 | - (stored kernel) | 0.664 (MIXED ours whole / arm kernel) | - | 2833.1 | 896.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 281.9 | 281.9..281.9 | 1 | 0.012 | - | - | 281.9 | - (stored kernel) | 0.012 (MIXED ours whole / arm kernel) | - | 2924.1 | 832.0 | rel_fro_vs_torch_eager_fp32=6.375e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on amd (DigitalOcean MI325X) job a1065

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3961.9 | 3961.9..3961.9 | 1 | - | - | 3961.9 | - | - (stored whole) | - | - | - | - | accuracy=0.920330 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 28098.7 | 28098.7..28098.7 | 1 | 0.141 | - | 28098.7 | 28098.7 | 0.00 (cpu-arm) | 0.141 (whole/whole) | - | 1143.5 | - | accuracy=0.910200 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### sgd-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1065)`, ran on amd (DigitalOcean MI325X) job a1065

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2320.4 | 2320.4..2320.4 | 1 | - | - | 2320.4 | - | - (stored whole) | - | - | - | - | accuracy=0.755330 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1065 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 8379.1 | 8379.1..8379.1 | 1 | 0.277 | - | 8379.1 | 8379.1 | 0.00 (cpu-arm) | 0.277 (whole/whole) | - | 266.7 | - | accuracy=0.752520 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on amd (DigitalOcean MI325X) job a0868

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 741.9 | 741.9..741.9 | 1 | - | - | 741.9 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=20.498972, relative_reconstruction_error_100k_rows=3.379e-05 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 2159.2 | 2159.2..2159.2 | 1 | 0.344 | - | - | 2159.2 | - (stored kernel) | 0.344 (MIXED ours whole / arm kernel) | - | 3796.1 | 3361.7 | max_rel_singular_value_error=2.707e+08, relative_reconstruction_error_100k_rows=0.026534 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 3741.4 | 3741.4..3741.4 | 1 | 0.198 | - | 3741.4 | 3741.4 | 0.00 (cpu-arm) | 0.198 (whole/whole) | - | 8708.9 | - | max_rel_singular_value_error=1.000000, relative_reconstruction_error_100k_rows=4.1e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0868)`, ran on amd (DigitalOcean MI325X) job a0868

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19.1 | 19.1..19.1 | 1 | - | - | 19.1 | - | - (stored whole) | - | - | - | - | max_rel_singular_value_error=7.036e-07, relative_reconstruction_error_100k_rows=1.18e-06 | - | main board, one scored run | - | ok (main@4df610f3b amd/a0868 2026-10-08; identity vs nvidia-l40s: MATCH) |
| torch-gpu | torch | gpu | opponent | 271.6 | 271.6..271.6 | 1 | 0.070 | - | - | 271.6 | - (stored kernel) | 0.070 (MIXED ours whole / arm kernel) | - | 2750.5 | 168.0 | max_rel_singular_value_error=4.401e-05, relative_reconstruction_error_100k_rows=0.003043 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| numpy-cpu | numpy | cpu | opponent | 130.5 | 130.5..130.5 | 1 | 0.146 | - | 130.5 | 130.5 | 0.00 (cpu-arm) | 0.146 (whole/whole) | - | 485.8 | - | max_rel_singular_value_error=4.308e-08, relative_reconstruction_error_100k_rows=4.314e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, numpy-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0495)`, ran on amd (DigitalOcean MI325X) job a0495

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1091.7 | 1091.7..1091.7 | 1 | - | - | 1091.7 | - | - (stored whole) | - | - | - | - | finite=True, r2=-0.106016, rmse=0.878373 | - | main board, one scored run | - | ok (main@89485bc96 amd/a0495 2026-10-08; identity vs nvidia-l40s: MATCH) |
| gpytorch-gpu | gpytorch | gpu | opponent | 48.7 | 48.7..48.7 | 1 | 22.398 | - | - | 48.7 | - (stored kernel) | 22.398 (MIXED ours whole / arm kernel) | - | 4591.1 | 1611.8 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from opponents-20261006; measured this run) |
| gpytorch-cpu | gpytorch | cpu | opponent | 672.0 | 672.0..672.0 | 1 | 1.625 | - | 672.0 | 672.0 | 0.00 (cpu-arm) | 1.625 (whole/whole) | - | 3457.1 | - | finite=True, r2=-0.106040, rmse=0.878383 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, gpytorch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, gpytorch-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows: None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 89.7 | 89.7..89.7 | 1 | - | - | 89.7 | - | - (stored whole) | - | - | - | - | output_shape=100000x8 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 202.8 | 202.8..202.8 | 1 | 0.442 | - | 202.8 | 202.8 | 0.00 (cpu-arm) | 0.442 (whole/whole) | - | 348.0 | - | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1068)`, ran on amd (DigitalOcean MI325X) job a1068

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 82.9 | 82.9..82.9 | 1 | - | - | 82.9 | - | - (stored whole) | - | - | - | - | output_shape=100000x5 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1068 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 116.5 | 116.5..116.5 | 1 | 0.712 | - | 116.5 | 116.5 | 0.00 (cpu-arm) | 0.712 (whole/whole) | - | 307.5 | - | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on amd (DigitalOcean MI325X) job a1069

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 189.2 | 189.2..189.2 | 1 | - | - | 189.2 | - | - (stored whole) | - | - | - | - | forecast_rmse=1.436610 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs nvidia-l40s: n/a) |
| statsforecast-cpu | statsforecast | cpu | opponent | 104.8 | 104.8..104.8 | 1 | 1.806 | - | 104.8 | 104.8 | 0.00 (cpu-arm) | 1.806 (whole/whole) | - | 283.6 | - | forecast_rmse=1.436557 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 38.6 | 38.6..38.6 | 1 | 4.903 | - | 38.6 | 38.6 | 0.00 (cpu-arm) | 4.903 (whole/whole) | - | 57.6 | - | forecast_rmse=1.434862 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### theta / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1069)`, ran on amd (DigitalOcean MI325X) job a1069

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 596.1 | 596.1..596.1 | 1 | - | - | 596.1 | - | - (stored whole) | - | - | - | - | forecast_rmse=49.020604 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1069 2026-10-09; identity vs nvidia-l40s: n/a) |
| statsforecast-cpu | statsforecast | cpu | opponent | 548.4 | 548.4..548.4 | 1 | 1.087 | - | 548.4 | 548.4 | 0.00 (cpu-arm) | 1.087 (whole/whole) | - | 283.4 | - | forecast_rmse=49.253901 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 52.3 | 52.3..52.3 | 1 | 11.399 | - | 52.3 | 52.3 | 0.00 (cpu-arm) | 11.399 (whole/whole) | - | 58.1 | - | forecast_rmse=49.311757 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, statsforecast-cpu, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'decomposition_type': 'multiplicative', 'season_length': 24}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / istella (rows full, shape X 100000x220; Xq 10000x220; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0877)`, ran on amd (DigitalOcean MI325X) job a0877

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.5 | 8.5..8.5 | 1 | - | - | 8.5 | - | - (stored whole) | - | - | - | - | max_additivity_error=1.175e-06 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0877 2026-10-08; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | opponent | 415.9 | 415.9..415.9 | 1 | 0.020 | - | 415.9 | 415.9 | 0.00 (cpu-arm) | 0.020 (whole/whole) | - | 1468.4 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 344.0 | 344.0..344.0 | 1 | 0.025 | - | 344.0 | 344.0 | 0.00 (cpu-arm) | 0.025 (whole/whole) | - | 1345.6 | - | max_additivity_error=1.837e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 275.4 | 275.4..275.4 | 1 | 0.031 | - | 275.4 | 275.4 | 0.00 (cpu-arm) | 0.031 (whole/whole) | - | 1532.8 | - | max_additivity_error=4.441e-15 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tree-shap / taxi (rows full, shape X 100000x11; Xq 10000x11; y 100000; yq 10000)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a0877)`, ran on amd (DigitalOcean MI325X) job a0877

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.8 | 2.8..2.8 | 1 | - | - | 2.8 | - | - (stored whole) | - | - | - | - | max_additivity_error=3.858e-05 | - | main board, one scored run | - | ok (main@42d1e42c6 amd/a0877 2026-10-08; identity vs nvidia-l40s: MATCH) |
| shap-cpu | shap | cpu | opponent | 161.7 | 161.7..161.7 | 1 | 0.018 | - | 161.7 | 161.7 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 399.3 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 156.1 | 156.1..156.1 | 1 | 0.018 | - | 156.1 | 156.1 | 0.00 (cpu-arm) | 0.018 (whole/whole) | - | 286.1 | - | max_additivity_error=0.0001201 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 117.1 | 117.1..117.1 | 1 | 0.024 | - | 117.1 | 117.1 | 0.00 (cpu-arm) | 0.024 (whole/whole) | - | 269.5 | - | max_additivity_error=5.684e-13 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, shap-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1067)`, ran on amd (DigitalOcean MI325X) job a1067

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1957.9 | 1957.9..1957.9 | 1 | - | - | 1957.9 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.992124 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1067 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 17896.2 | 17896.2..17896.2 | 1 | 0.109 | - | 17896.2 | 17896.2 | 0.00 (cpu-arm) | 0.109 (whole/whole) | - | 370.6 | - | trustworthiness_k15=0.992170 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters: not checked on the main board (our one scored run; opponents copied, never re-run)

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc None, log `/Users/andrewhendel/mojolearn-evidence/grid-lq/amd-results.txt (amd/a1067)`, ran on amd (DigitalOcean MI325X) job a1067

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | whole ms | kernel ms | copy ms (source) | ours IDENTICAL / arm (clock) | ours FAST / arm (clock) | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1902.7 | 1902.7..1902.7 | 1 | - | - | 1902.7 | - | - (stored whole) | - | - | - | - | trustworthiness_k15=0.998921 | - | main board, one scored run | - | ok (main@8ed94710a amd/a1067 2026-10-09; identity vs nvidia-l40s: n/a) |
| sklearn-cpu | scikit-learn | cpu | opponent | 15382.8 | 15382.8..15382.8 | 1 | 0.124 | - | 15382.8 | 15382.8 | 0.00 (cpu-arm) | 0.124 (whole/whole) | - | 379.6 | - | trustworthiness_k15=0.998823 | - | LIKE-FOR-LIKE-SPAN | - | ok (copied from release-board-resume-r2; measured this run) |

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

