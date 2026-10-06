# mojolearn benchmark board

Generated 2026-10-06T07:53:57Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | nvidia / cuda |
| GPU | NVIDIA L40S |
| GPU driver | 580.167.08 |
| CPU | AMD EPYC 9554 64-Core Processor (128 logical cores) |
| memory bytes | 1081851899904 |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.10 CPython |
| mojolearn | None (wheel None, sha256 None) |
| script commit | - |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, cuml-cu12 26.8.0, cuvs-cu12 26.8.1, torch 2.13.0+cu129, numba 0.64.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.4.6 |

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

Races: 113 planned, 71 done, 2 failed, 0 unsupported, 40 pending. Cells: 163 (REFUSED 2, ok 161).

Inference cells: 132 (REFUSED 2, ok 130).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adagrad | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adamax | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.395e-09 |
| algos | als | taxi-zones | recall_at_10 (higher is better) | - | - | implicit-gpu 0.076266 |
| algos | autoarima | synthetic | forecast_rmse (lower is better) | - | - | cuml-gpu 32.552990 |
| algos | avgpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002792; torch-eager-tf32 0.000000; torch-compile-tf32 0.002792; torch-eager-bf16 0.000000; torch-compile-bf16 0.002792 |
| algos | batchnorm1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.109e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.109e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.109e-08 |
| algos | bernoulli-nb | istella | accuracy (higher is better) | - | - | cuml-gpu 0.794050 |
| algos | bernoulli-nb | istella | logloss (lower is better) | - | - | cuml-gpu 5.350631 |
| algos | categorical-nb | istella | accuracy (higher is better) | - | - | cuml-gpu 0.838850 |
| algos | categorical-nb | istella | logloss (lower is better) | - | - | cuml-gpu 0.412625 |
| algos | cholesky | synthetic | relative_residual | - | - | torch-gpu 1.509e-07; cupy-gpu 1.365e-07 |
| algos | cnn-clf | synthetic | accuracy (higher is better) | - | - | torch-eager-fp32 1.000000; torch-compile-fp32 1.000000; torch-eager-tf32 1.000000; torch-compile-tf32 1.000000; torch-eager-bf16 1.000000; torch-compile-bf16 1.000000 |
| algos | complement-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.678060 |
| algos | complement-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 0.715531 |
| algos | connected-components | istella | n_components | - | - | cugraph-gpu 81 |
| algos | conv1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 3417.849541; torch-compile-bf16 3524.661064 |
| algos | conv1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.003296; torch-compile-bf16 0.003294 |
| algos | cross-entropy | synthetic | loss_rel_err_vs_fp64 | - | - | torch-eager-fp32 5.477e-08; torch-compile-fp32 4.564e-08 |
| algos | dart-reg | taxi | r2 (higher is better) | - | - | xgboost-gpu 0.925745 |
| algos | dart-reg | taxi | rmse (lower is better) | - | - | xgboost-gpu 4.340112 |
| algos | dart | taxi | accuracy (higher is better) | - | - | xgboost-gpu 0.768030 |
| algos | dart | taxi | logloss (lower is better) | - | - | xgboost-gpu 0.529584 |
| algos | decision-tree-clf | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.756300 |
| algos | decision-tree-clf | taxi | logloss (lower is better) | - | - | cuml-gpu 1.202243 |
| algos | decision-tree-reg | taxi | r2 (higher is better) | - | - | cuml-gpu 0.861978 |
| algos | decision-tree-reg | taxi | rmse (lower is better) | - | - | cuml-gpu 5.917127 |
| algos | eigh | synthetic | max_eigenvalue_error | - | - | torch-gpu 1.044e-06; cupy-gpu 1.044e-06 |
| algos | eigh | synthetic | relative_residual | - | - | torch-gpu 1.016e-06; cupy-gpu 1.016e-06 |
| algos | gaussian-nb | istella | accuracy (higher is better) | - | - | cuml-gpu 0.876570 |
| algos | gaussian-nb | istella | logloss (lower is better) | - | - | cuml-gpu 3.416741 |
| algos | gaussian-rp | istella | mean_abs_distortion | - | - | cuml-gpu 0.443920 |
| algos | gcn | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.009646; torch-eager-tf32 281.122146; torch-compile-tf32 281.117866; torch-eager-bf16 2718.059111; torch-compile-bf16 2718.055850 |
| algos | gcn | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.041e-07; torch-eager-tf32 0.0002661; torch-compile-tf32 0.0002661; torch-eager-bf16 0.002187; torch-compile-bf16 0.002187 |
| algos | global-avgpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.009731; torch-eager-tf32 0.000000; torch-compile-tf32 0.009731; torch-eager-bf16 0.000000; torch-compile-bf16 0.009731 |
| algos | global-avgpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.64e-08; torch-eager-tf32 0.000000; torch-compile-tf32 9.64e-08; torch-eager-bf16 0.000000; torch-compile-bf16 9.64e-08 |
| algos | graphsage | istella | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.134110; torch-eager-tf32 430.934131; torch-compile-tf32 430.934131; torch-eager-bf16 3907.114267; torch-compile-bf16 3678.210080 |
| algos | graphsage | istella | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.7e-08; torch-eager-tf32 0.0002969; torch-compile-tf32 0.0002969; torch-eager-bf16 0.003366; torch-compile-bf16 0.003054 |
| algos | gru-clf | synthetic | accuracy (higher is better) | - | - | torch-eager-fp32 0.971842; torch-compile-fp32 0.971842; torch-eager-tf32 0.971842; torch-compile-tf32 0.971842; torch-eager-bf16 0.971951; torch-compile-bf16 0.971951 |
| algos | gru-reg | synthetic | r2 (higher is better) | - | - | torch-eager-fp32 0.981946; torch-compile-fp32 0.981946; torch-eager-tf32 0.981946; torch-compile-tf32 0.981946; torch-eager-bf16 0.981898; torch-compile-bf16 0.981898 |
| algos | gru-reg | synthetic | rmse (lower is better) | - | - | torch-eager-fp32 0.155672; torch-compile-fp32 0.155672; torch-eager-tf32 0.155672; torch-compile-tf32 0.155672; torch-eager-bf16 0.155878; torch-compile-bf16 0.155878 |
| algos | incremental-pca | istella | explained_variance_fraction | - | - | cuml-gpu 1.000000 |
| algos | ivf-pq | istella | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.791300 |
| algos | ivf-refine | istella | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.993625 |
| algos | ivf-sq | istella | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.469725 |
| algos | kernel-shap | istella | rel_error_vs_exact | - | - | cuml-gpu 0.032177 |
| algos | lars | istella | r2 (higher is better) | - | - | cuml-gpu 0.328088 |
| algos | lars | istella | rmse (lower is better) | - | - | cuml-gpu 0.684726 |
| algos | layernorm | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.006419; torch-eager-tf32 0.000000; torch-compile-tf32 0.006419; torch-eager-bf16 0.000000; torch-compile-bf16 0.006419 |
| algos | layernorm | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.316e-08; torch-eager-tf32 0.000000; torch-compile-tf32 5.316e-08; torch-eager-bf16 0.000000; torch-compile-bf16 5.316e-08 |
| algos | louvain | taxi | modularity | - | - | cugraph-gpu 0.941795 |
| algos | louvain | taxi | n_communities | - | - | cugraph-gpu 62 |
| algos | lstm-clf | taxi-hourly | accuracy (higher is better) | - | - | torch-eager-fp32 0.868218; torch-compile-fp32 0.868218; torch-eager-tf32 0.868164; torch-compile-tf32 0.868164; torch-eager-bf16 0.868327; torch-compile-bf16 0.868327 |
| algos | lstm-reg | taxi-hourly | r2 (higher is better) | - | - | torch-eager-fp32 0.751679; torch-compile-fp32 0.751679; torch-eager-tf32 0.751680; torch-compile-tf32 0.751680; torch-eager-bf16 0.751712; torch-compile-bf16 0.751712 |
| algos | lstm-reg | taxi-hourly | rmse (lower is better) | - | - | torch-eager-fp32 0.540430; torch-compile-fp32 0.540430; torch-eager-tf32 0.540429; torch-compile-tf32 0.540429; torch-eager-bf16 0.540394; torch-compile-bf16 0.540394 |
| algos | lstsq | taxi | relative_residual | - | - | torch-gpu 0.756366; cupy-gpu 0.756366 |
| algos | lu-solve | synthetic | relative_residual | - | - | torch-gpu 3.386e-07; cupy-gpu 3.386e-07 |
| algos | maxpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | multinomial-nb | istella | accuracy (higher is better) | - | - | cuml-gpu 0.853620 |
| algos | multinomial-nb | istella | logloss (lower is better) | - | - | cuml-gpu 3.628599 |
| algos | multinomial-nb | text | accuracy (higher is better) | - | - | cuml-gpu 0.983067 |
| algos | multinomial-nb | text | logloss (lower is better) | - | - | cuml-gpu 0.559524 |
| algos | pagerank | istella | sum | - | - | cugraph-gpu 1.000000 |
| algos | permutation-shap | istella | rel_error_vs_exact | - | - | cuml-gpu 1.483e-07 |
| algos | qn-reg | istella | r2 (higher is better) | - | - | cuml-gpu 0.327567 |
| algos | qn-reg | istella | rmse (lower is better) | - | - | cuml-gpu 0.684991 |
| algos | qr | istella | relative_gram_difference | - | - | torch-gpu 3.787e-07; cupy-gpu 3.787e-07 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | - | - | torch-gpu 0.0002359 |
| algos | resnet-block | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.818182; torch-eager-tf32 1532.793045; torch-compile-tf32 1532.793045; torch-eager-bf16 17838.627100; torch-compile-bf16 18497.318029 |
| algos | resnet-block | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.381e-07; torch-eager-tf32 0.0003478; torch-compile-tf32 0.0003478; torch-eager-bf16 0.003621; torch-compile-bf16 0.003425 |
| algos | rnn-clf | synthetic | accuracy (higher is better) | - | - | torch-eager-fp32 0.953559; torch-compile-fp32 0.953559; torch-eager-tf32 0.953505; torch-compile-tf32 0.953505; torch-eager-bf16 0.953559; torch-compile-bf16 0.953559 |
| algos | rnn-reg | synthetic | r2 (higher is better) | - | - | torch-eager-fp32 0.977347; torch-compile-fp32 0.977347; torch-eager-tf32 0.977347; torch-compile-tf32 0.977347; torch-eager-bf16 0.977345; torch-compile-bf16 0.977345 |
| algos | rnn-reg | synthetic | rmse (lower is better) | - | - | torch-eager-fp32 0.174374; torch-compile-fp32 0.174374; torch-eager-tf32 0.174374; torch-compile-tf32 0.174374; torch-eager-bf16 0.174385; torch-compile-bf16 0.174385 |
| algos | sgd-clf | istella | accuracy (higher is better) | - | - | cuml-gpu 0.809350 |
| algos | sgd-reg | istella | r2 (higher is better) | - | - | cuml-gpu 0.327768 |
| algos | sgd-reg | istella | rmse (lower is better) | - | - | cuml-gpu 0.684889 |
| algos | sgd | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.772e-10 |
| algos | simple-imputer | taxi | masked_rmse | - | - | cuml-gpu 5.985180 |
| algos | sparse-rp | taxi | mean_abs_distortion | - | - | cuml-gpu 0.266849 |
| algos | svd | taxi | max_rel_singular_value_error | - | - | torch-gpu 2.595e-06; cupy-gpu 3.763e-06 |
| algos | svd | taxi | relative_reconstruction_error_100k_rows | - | - | torch-gpu 7.244e-06; cupy-gpu 6.886e-06 |
| algos | svgp | taxi | r2 (higher is better) | - | - | gpytorch-gpu -0.209527 |
| algos | svgp | taxi | rmse (lower is better) | - | - | gpytorch-gpu 17.831018 |
| algos | tree-shap | taxi | max_additivity_error | - | - | xgboost-gpu 5.402e-05 |
| algos | tsne | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | - | cuml-gpu 0.998353 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| algos | avgpool1d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm -); torch-compile-fp32 0.4 ms (IDENTICAL/arm -); torch-eager-tf32 0.2 ms (IDENTICAL/arm -); torch-compile-tf32 0.3 ms (IDENTICAL/arm -); torch-eager-bf16 0.2 ms (IDENTICAL/arm -); torch-compile-bf16 0.3 ms (IDENTICAL/arm -) |
| algos | batchnorm1d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.3 ms (IDENTICAL/arm -); torch-compile-fp32 0.4 ms (IDENTICAL/arm -); torch-eager-tf32 0.3 ms (IDENTICAL/arm -); torch-compile-tf32 0.4 ms (IDENTICAL/arm -); torch-eager-bf16 0.3 ms (IDENTICAL/arm -); torch-compile-bf16 0.4 ms (IDENTICAL/arm -) |
| algos | bernoulli-nb | istella | Xq | - | - | - | - | cuml-gpu 9.8 ms (IDENTICAL/arm -) |
| algos | binarizer | istella | Xq | - | - | - | - | cuml-gpu 9.1 ms (IDENTICAL/arm -) |
| algos | cagra | istella | Xq | - | - | - | - | cuvs-gpu - ms (IDENTICAL/arm -) |
| algos | categorical-nb | istella | Xq | - | - | - | - | cuml-gpu 6.6 ms (IDENTICAL/arm -) |
| algos | cnn-clf | synthetic | Xq | - | - | - | - | torch-eager-fp32 1.7 ms (IDENTICAL/arm -); torch-compile-fp32 1.4 ms (IDENTICAL/arm -); torch-eager-tf32 1.7 ms (IDENTICAL/arm -); torch-compile-tf32 1.3 ms (IDENTICAL/arm -); torch-eager-bf16 1.4 ms (IDENTICAL/arm -); torch-compile-bf16 1.5 ms (IDENTICAL/arm -) |
| algos | complement-nb | taxi | Xq | - | - | - | - | cuml-gpu 1.4 ms (IDENTICAL/arm -) |
| algos | conv1d | synthetic | Xq | - | - | - | - | torch-eager-fp32 1.2 ms (IDENTICAL/arm -); torch-compile-fp32 1.5 ms (IDENTICAL/arm -); torch-eager-tf32 1.2 ms (IDENTICAL/arm -); torch-compile-tf32 1.3 ms (IDENTICAL/arm -); torch-eager-bf16 1.1 ms (IDENTICAL/arm -); torch-compile-bf16 1.1 ms (IDENTICAL/arm -) |
| algos | dart-reg | taxi | Xq | - | - | - | - | xgboost-gpu 13.8 ms (IDENTICAL/arm -) |
| algos | dart | taxi | Xq | - | - | - | - | xgboost-gpu 15.0 ms (IDENTICAL/arm -) |
| algos | decision-tree-clf | taxi | Xq | - | - | - | - | cuml-gpu 0.8 ms (IDENTICAL/arm -) |
| algos | decision-tree-reg | taxi | Xq | - | - | - | - | cuml-gpu 0.4 ms (IDENTICAL/arm -) |
| algos | gaussian-nb | istella | Xq | - | - | - | - | cuml-gpu 45.3 ms (IDENTICAL/arm -) |
| algos | gaussian-rp | istella | Xq | - | - | - | - | cuml-gpu 11.6 ms (IDENTICAL/arm -) |
| algos | gcn | istella | Xq | - | - | - | - | torch-eager-fp32 6.9 ms (IDENTICAL/arm -); torch-compile-fp32 1.9 ms (IDENTICAL/arm -); torch-eager-tf32 6.9 ms (IDENTICAL/arm -); torch-compile-tf32 2.1 ms (IDENTICAL/arm -); torch-eager-bf16 5.4 ms (IDENTICAL/arm -); torch-compile-bf16 2.1 ms (IDENTICAL/arm -) |
| algos | global-avgpool | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-tf32 0.1 ms (IDENTICAL/arm -); torch-compile-tf32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |
| algos | graphsage | istella | Xq | - | - | - | - | torch-eager-fp32 7.7 ms (IDENTICAL/arm -); torch-compile-fp32 3.2 ms (IDENTICAL/arm -); torch-eager-tf32 7.5 ms (IDENTICAL/arm -); torch-compile-tf32 2.8 ms (IDENTICAL/arm -); torch-eager-bf16 7.4 ms (IDENTICAL/arm -); torch-compile-bf16 2.6 ms (IDENTICAL/arm -) |
| algos | gru-clf | synthetic | Xq | - | - | - | - | torch-eager-fp32 2.9 ms (IDENTICAL/arm -); torch-compile-fp32 3.2 ms (IDENTICAL/arm -); torch-eager-tf32 2.7 ms (IDENTICAL/arm -); torch-compile-tf32 3.2 ms (IDENTICAL/arm -); torch-eager-bf16 5.2 ms (IDENTICAL/arm -); torch-compile-bf16 3.7 ms (IDENTICAL/arm -) |
| algos | gru-reg | synthetic | Xq | - | - | - | - | torch-eager-fp32 3.0 ms (IDENTICAL/arm -); torch-compile-fp32 3.2 ms (IDENTICAL/arm -); torch-eager-tf32 2.6 ms (IDENTICAL/arm -); torch-compile-tf32 3.2 ms (IDENTICAL/arm -); torch-eager-bf16 3.4 ms (IDENTICAL/arm -); torch-compile-bf16 3.6 ms (IDENTICAL/arm -) |
| algos | incremental-pca | istella | Xq | - | - | - | - | cuml-gpu 6.1 ms (IDENTICAL/arm -) |
| algos | ivf-pq | istella | Xq | - | - | - | - | cuvs-gpu 11.8 ms (IDENTICAL/arm -) |
| algos | ivf-refine | istella | Xq | - | - | - | - | cuvs-gpu 82.4 ms (IDENTICAL/arm -) |
| algos | ivf-sq | istella | Xq | - | - | - | - | cuvs-gpu 6.0 ms (IDENTICAL/arm -) |
| algos | kbins | istella | Xq | - | - | - | - | cuml-gpu 26.0 ms (IDENTICAL/arm -) |
| algos | label-binarizer | istella | Xq | - | - | - | - | cuml-gpu 6.8 ms (IDENTICAL/arm -) |
| algos | label-encoder | istella | Xq | - | - | - | - | cuml-gpu 3.2 ms (IDENTICAL/arm -) |
| algos | lars | istella | Xq | - | - | - | - | cuml-gpu 6.3 ms (IDENTICAL/arm -) |
| algos | layernorm | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm -); torch-compile-fp32 0.8 ms (IDENTICAL/arm -); torch-eager-tf32 0.2 ms (IDENTICAL/arm -); torch-compile-tf32 0.3 ms (IDENTICAL/arm -); torch-eager-bf16 0.2 ms (IDENTICAL/arm -); torch-compile-bf16 0.3 ms (IDENTICAL/arm -) |
| algos | lstm-clf | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 3.2 ms (IDENTICAL/arm -); torch-compile-fp32 3.6 ms (IDENTICAL/arm -); torch-eager-tf32 3.2 ms (IDENTICAL/arm -); torch-compile-tf32 3.6 ms (IDENTICAL/arm -); torch-eager-bf16 5.7 ms (IDENTICAL/arm -); torch-compile-bf16 6.0 ms (IDENTICAL/arm -) |
| algos | lstm-reg | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 3.4 ms (IDENTICAL/arm -); torch-compile-fp32 3.5 ms (IDENTICAL/arm -); torch-eager-tf32 3.2 ms (IDENTICAL/arm -); torch-compile-tf32 3.3 ms (IDENTICAL/arm -); torch-eager-bf16 6.4 ms (IDENTICAL/arm -); torch-compile-bf16 6.7 ms (IDENTICAL/arm -) |
| algos | maxabs-scaler | taxi | Xq | - | - | - | - | cuml-gpu 0.9 ms (IDENTICAL/arm -) |
| algos | maxpool2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.6 ms (IDENTICAL/arm -); torch-compile-fp32 0.5 ms (IDENTICAL/arm -); torch-eager-tf32 0.6 ms (IDENTICAL/arm -); torch-compile-tf32 0.5 ms (IDENTICAL/arm -); torch-eager-bf16 0.6 ms (IDENTICAL/arm -); torch-compile-bf16 0.6 ms (IDENTICAL/arm -) |
| algos | minmax-scaler | taxi | Xq | - | - | - | - | cuml-gpu 0.7 ms (IDENTICAL/arm -) |
| algos | multinomial-nb | istella | Xq | - | - | - | - | cuml-gpu 2.2 ms (IDENTICAL/arm -) |
| algos | multinomial-nb | text | Xq | - | - | - | - | cuml-gpu 2.4 ms (IDENTICAL/arm -) |
| algos | normalizer | istella | Xq | - | - | - | - | cuml-gpu 1.6 ms (IDENTICAL/arm -) |
| algos | onehot | istella | Xq | - | - | - | - | cuml-gpu 54.8 ms (IDENTICAL/arm -) |
| algos | poly-features | istella | Xq | - | - | - | - | cuml-gpu 5.9 ms (IDENTICAL/arm -) |
| algos | power-transformer | istella | Xq | - | - | - | - | cuml-gpu - ms (IDENTICAL/arm -) |
| algos | quantile-transformer | istella | Xq | - | - | - | - | cuml-gpu 282.0 ms (IDENTICAL/arm -) |
| algos | resnet-block | synthetic | Xq | - | - | - | - | torch-eager-fp32 1.4 ms (IDENTICAL/arm -); torch-compile-fp32 1.9 ms (IDENTICAL/arm -); torch-eager-tf32 1.1 ms (IDENTICAL/arm -); torch-compile-tf32 1.0 ms (IDENTICAL/arm -); torch-eager-bf16 0.9 ms (IDENTICAL/arm -); torch-compile-bf16 0.7 ms (IDENTICAL/arm -) |
| algos | rnn-clf | synthetic | Xq | - | - | - | - | torch-eager-fp32 1.9 ms (IDENTICAL/arm -); torch-compile-fp32 3.1 ms (IDENTICAL/arm -); torch-eager-tf32 1.9 ms (IDENTICAL/arm -); torch-compile-tf32 3.2 ms (IDENTICAL/arm -); torch-eager-bf16 1.8 ms (IDENTICAL/arm -); torch-compile-bf16 2.0 ms (IDENTICAL/arm -) |
| algos | rnn-reg | synthetic | Xq | - | - | - | - | torch-eager-fp32 2.8 ms (IDENTICAL/arm -); torch-compile-fp32 2.1 ms (IDENTICAL/arm -); torch-eager-tf32 2.8 ms (IDENTICAL/arm -); torch-compile-tf32 3.3 ms (IDENTICAL/arm -); torch-eager-bf16 4.8 ms (IDENTICAL/arm -); torch-compile-bf16 4.4 ms (IDENTICAL/arm -) |
| algos | robust-scaler | istella | Xq | - | - | - | - | cuml-gpu 4.2 ms (IDENTICAL/arm -) |
| algos | sgd-clf | istella | Xq | - | - | - | - | cuml-gpu 0.8 ms (IDENTICAL/arm -) |
| algos | sgd-reg | istella | Xq | - | - | - | - | cuml-gpu 0.7 ms (IDENTICAL/arm -) |
| algos | simple-imputer | taxi | Xq | - | - | - | - | cuml-gpu 3.3 ms (IDENTICAL/arm -) |
| algos | sparse-rp | taxi | Xq | - | - | - | - | cuml-gpu 4.6 ms (IDENTICAL/arm -) |
| algos | standard-scaler | taxi | Xq | - | - | - | - | cuml-gpu 0.7 ms (IDENTICAL/arm -) |
| algos | svgp | taxi | Xq | - | - | - | - | gpytorch-gpu 7.2 ms (IDENTICAL/arm -) |
| algos | target-encoder | taxi | Xq | - | - | - | - | cuml-gpu 29.8 ms (IDENTICAL/arm -) |

## Algorithm expansion

### adagrad / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adagrad.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 14.9 | 14.9..14.9 | 1 | - | - | 968.5 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 115.6 | 115.6..115.6 | 1 | - | - | 1302.1 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'eps': 1e-10, 'initial_accumulator_value': 0.0, 'lr': 0.001, 'lr_decay': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| eps | 1e-10 | 1e-10 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### adamax / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adamax.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 20.2 | 20.2..20.2 | 1 | - | - | 981.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 146.9 | 146.9..146.9 | 1 | - | - | 1445.6 | 896.0 | rel_fro_vs_torch_eager_fp32=6.395e-09 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| betas | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### als / taxi-zones (rows full, shape X 129352x261; Xq 14373x261)

race: done, driver rc 0, log `logs/algos.als.taxi-zones.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implicit-gpu | implicit | gpu | opponent | 879.9 | 879.9..879.9 | 1 | - | - | 649.8 | 514.0 | recall_at_10=0.076266 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, implicit-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'calculate_training_loss': False, 'cg_steps': 3, 'factors': 64, 'iterations': 15, 'random_state': 7, 'regularization': 0.01, 'use_cg': False}. Rows: None. Timed: None.

mismatch: implicit-gpu has only the conjugate-gradient solver (use_cg ignored there); ours and implicit-cpu solve each least-squares step exactly (use_cg=False)

mismatch: each library draws its own initial factors from random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `implicit-gpu`, seed 7): MATCHED

| parameter | implicit-gpu |
|---|---|
| library (source) | implicit (declared) |
| alpha | 1.0 |
| n_estimators | 15 |
| seed | 7 |

### autoarima / synthetic (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.autoarima.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 822483.3 | 822483.3..822483.3 | 1 | - | - | 860.9 | 810.0 | forecast_rmse=32.552990 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'allow_intercept': True, 'ic': 'aicc', 'max_d': 1, 'max_p': 3, 'max_q': 3, 'seasonal': False, 'stepwise': False}. Rows: None. Timed: None.

mismatch: the likelihood optimizer and its stopping rule are each library's own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (declared) |
| seasonal | false |
| seed | "none (deterministic)" |

### avgpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool1d.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 2.5 | 2.5..2.5 | 1 | - | - | 695.3 | 224.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.4 | 2.4..2.4 | 1 | - | - | 924.1 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 695.8 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | - | - | 875.6 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | - | - | 695.5 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.5 | 2.5..2.5 | 1 | - | - | 870.2 | 224.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm1d.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 776.4 | 320.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 982.0 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 2.5 | 2.5..2.5 | 1 | - | - | 776.6 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.2 | 2.2..2.2 | 1 | - | - | 917.4 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | - | - | 776.4 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | - | - | 910.1 | 320.0 | max_rel_diff_vs_torch_eager_fp32=0.002792, rel_fro_vs_torch_eager_fp32=5.109e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 256, 'track_running_stats': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| momentum | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bernoulli-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bernoulli-nb.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 64.3 | 64.3..64.3 | 1 | - | - | 3129.9 | 1370.0 | accuracy=0.794050, logloss=5.350631 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), BernoulliNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 9.8 | 9.8..9.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### binarizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.binarizer.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 2.5 | 2.5..2.5 | 1 | - | - | 3009.1 | 1440.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'threshold': 0.0}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Binarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 9.1 | 9.1..9.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### cagra / istella (rows full, shape index 400000x220; queries 4000x220)

race: failed, driver rc 1, log `logs/algos.cagra.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | - | - | 0 | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "CuvsException('RAFT failure at file=/__w/cuvs/cuvs/cpp/src/neighbors/detail/cagra/graph_core.cuh line=1679: Could not generate an intermediate CAGRA graph because the initial kNN graph cont) (measured this run) |

settings: {'graph_degree': 32, 'intermediate_graph_degree': 64, 'itopk_size': 64, 'n_neighbors': 10, 'random_state': 7}. Rows: None. Timed: None.

mismatch: faiss-cpu is HNSW (IndexHNSWFlat M=32, efConstruction=128, efSearch=64), the CPU graph index; cuvs-gpu is CAGRA itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuvs-gpu`, seed 7): MATCHED

| parameter | cuvs-gpu |
|---|---|
| library (source) | cuvs (declared) |
| n_neighbors | 10 |
| seed | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: cuvs-gpu seed: cuVS IndexParams take no seed and the index build samples rows; ours and faiss get 7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | Xq | - | - | - | 0 | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "CuvsException('RAFT failure at file=/__w/cuvs/cuvs/cpp/src/neighbors/detail/cagra/graph_core.cuh line=1679: Could not generate an intermediate CAGRA graph because the initial kNN graph cont) |

inference call, cuvs-gpu: search(queries)(Xq)

### categorical-nb / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.categorical-nb.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 76.1 | 76.1..76.1 | 1 | - | - | 1007.4 | 476.0 | accuracy=0.838850, logloss=0.412625 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

mismatch: min_categories = every code seen in X or Xq on ours and scikit-learn; cuML has no min_categories option

config: cuML benchmark (RAPIDS), CategoricalNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 6.6 | 6.6..6.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### cholesky / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cholesky.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 14.0 | 14.0..14.0 | 1 | - | - | 1214.5 | 768.3 | relative_residual=1.509e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 14.9 | 14.9..14.9 | 1 | - | - | 1285.6 | 1234.0 | relative_residual=1.365e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'jitter': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### cnn-clf / synthetic (rows full, shape X 20000x1x28x28; Xq 5000x1x28x28; y 20000; yq 5000)

race: done, driver rc 0, log `logs/algos.cnn-clf.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 458.6 | 458.6..458.6 | 1 | - | - | 1317.2 | 287.8 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 569.3 | 569.3..569.3 | 1 | - | - | 1531.3 | 214.3 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 475.9 | 475.9..475.9 | 1 | - | - | 1318.3 | 287.8 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 464.8 | 464.8..464.8 | 1 | - | - | 1358.0 | 214.3 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 483.3 | 483.3..483.3 | 1 | - | - | 1552.4 | 204.7 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 683.3 | 683.3..683.3 | 1 | - | - | 1618.5 | 154.0 | accuracy=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 128, 'conv_channels': [8, 16], 'dampening': 0.0, 'input_shape': [1, 28, 28], 'kernel_size': 3, 'learning_rate': 0.01, 'max_iter': 2, 'momentum': 0.9, 'nesterov': False, 'optimizer': 'sgd', 'pool_size': 2, 'random_state': 7, 'shuffle': True, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 128 | 128 | 128 | 128 | 128 | 128 |
| dampening | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| learning_rate | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| max_iter | 2 | 2 | 2 | 2 | 2 | 2 |
| momentum | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| nesterov | false | false | false | false | false | false |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 1.7 | 1.7..1.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 1.7 | 1.7..1.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 1.5 | 1.5..1.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### complement-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.complement-nb.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 12.3 | 12.3..12.3 | 1 | - | - | 1100.8 | 496.0 | accuracy=0.678060, logloss=0.715531 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True, 'norm': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), ComplementNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### connected-components / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.connected-components.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cugraph-gpu | cugraph | gpu | opponent | 19.4 | 19.4..19.4 | 1 | - | - | 959.5 | 432.0 | n_components=81 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cugraph-gpu`, seed 7): MATCHED

| parameter | cugraph-gpu |
|---|---|
| library (source) | cugraph (declared) |
| seed | "none (deterministic)" |

### conv1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv1d.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 3.8 | 3.8..3.8 | 1 | - | - | 989.0 | 640.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 5.1 | 5.1..5.1 | 1 | - | - | 1180.9 | 640.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 3.9 | 3.9..3.9 | 1 | - | - | 1005.1 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 4.1 | 4.1..4.1 | 1 | - | - | 1148.7 | 900.4 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 3.3 | 3.3..3.3 | 1 | - | - | 979.7 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3417.849541, rel_fro_vs_torch_eager_fp32=0.003296 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | - | - | 1181.1 | 706.5 | max_rel_diff_vs_torch_eager_fp32=3524.661064, rel_fro_vs_torch_eager_fp32=0.003294 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 128, 'kernel_size': 3, 'out_channels': 128, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 1.2 | 1.2..1.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.5 | 1.5..1.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 1.2 | 1.2..1.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### cross-entropy / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.cross-entropy.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 7.5 | 7.5..7.5 | 1 | - | - | 839.4 | 1024.1 | loss_rel_err_vs_fp64=5.477e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 6.4 | 6.4..6.4 | 1 | - | - | 1076.0 | 512.1 | loss_rel_err_vs_fp64=4.564e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ignore_index': -100, 'label_smoothing': 0.0, 'reduction': 'mean'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

### dart-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart-reg.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 1740.5 | 1740.5..1740.5 | 1 | - | - | 516.3 | 500.0 | finite=True, r2=0.925745, rmse=4.340112 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `xgboost-gpu`, seed 7): MATCHED

| parameter | xgboost-gpu |
|---|---|
| library (source) | xgboost (get_params) |
| boosting_type | "dart" |
| feature_fraction | 1.0 |
| feature_fraction_bynode | null |
| grow_policy | null |
| learning_rate | 0.1 |
| max_bin | 255 |
| max_depth | 8 |
| max_leaves | null |
| min_child_weight | 1.0 |
| min_split_gain | null |
| n_estimators | 200 |
| reg_alpha | 0.0 |
| reg_lambda | 0.0 |
| scale_pos_weight | 1.0 |
| seed | 7 |
| subsample | 1.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | Xq | - | 13.8 | 13.8..13.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, xgboost-gpu: predict(Xq)(Xq)

### dart / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 1701.0 | 1701.0..1701.0 | 1 | - | - | 522.5 | 500.0 | accuracy=0.768030, logloss=0.529584 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'bagging_seed': 7, 'colsample_bytree': 1.0, 'drop_rate': 0.1, 'drop_seed': 7, 'feature_fraction_seed': 7, 'learning_rate': 0.1, 'max_bin': 255, 'max_delta_step': 0.0, 'max_depth': 8, 'max_drop': 50, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 255, 'random_state': 7, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'skip_drop': 0.5, 'subsample': 1.0, 'subsample_freq': 0, 'uniform_drop': False, 'xgboost_dart_mode': False}. Rows: None. Timed: None.

mismatch: LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)

mismatch: XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM 20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode (XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come from random_state=7; each library's drop RNG is its own

mismatch: max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `xgboost-gpu`, seed 7): MATCHED

| parameter | xgboost-gpu |
|---|---|
| library (source) | xgboost (get_params) |
| boosting_type | "dart" |
| feature_fraction | 1.0 |
| feature_fraction_bynode | null |
| grow_policy | null |
| learning_rate | 0.1 |
| max_bin | 255 |
| max_depth | 8 |
| max_leaves | null |
| min_child_weight | 1.0 |
| min_split_gain | null |
| n_estimators | 200 |
| reg_alpha | 0.0 |
| reg_lambda | 0.0 |
| scale_pos_weight | 1.0 |
| seed | 7 |
| subsample | 1.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | Xq | - | 15.0 | 15.0..15.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, xgboost-gpu: predict(Xq)(Xq)

### decision-tree-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-clf.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 36.9 | 36.9..36.9 | 1 | - | - | 1000.0 | 492.0 | accuracy=0.756300, logloss=1.202243 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 16, 'max_features': 1.0, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree, no bootstrap, every feature (no GPU single-tree class exists), n_bins=128 as ours

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| bootstrap | false |
| class_weight | null |
| max_bin | 128 |
| max_depth | 16 |
| max_features | 1.0 |
| max_leaves | -1 |
| max_samples | 1.0 |
| min_samples_leaf | 1 |
| min_split_gain | 0.0 |
| n_estimators | 1 |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.8 | 0.8..0.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### decision-tree-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-reg.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 46.1 | 46.1..46.1 | 1 | - | - | 914.6 | 488.0 | finite=True, r2=0.861978, rmse=5.917127 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 16, 'max_features': 1.0, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 7, 'splitter': 'best'}. Rows: None. Timed: None.

mismatch: cuml-gpu is cuML's forest with one tree (see decision-tree-clf)

mismatch: ours' DecisionTree* is its forest builder with one tree: it splits on n_bins quantile bins per feature (128; 256 in FAST, rf_dt_default_bins) (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| bootstrap | false |
| max_bin | 128 |
| max_depth | 16 |
| max_features | 1.0 |
| max_leaves | -1 |
| max_samples | 1.0 |
| min_samples_leaf | 1 |
| min_split_gain | 0.0 |
| n_estimators | 1 |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### eigh / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.eigh.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 81.2 | 81.2..81.2 | 1 | - | - | 937.9 | 450.1 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 80.4 | 80.4..80.4 | 1 | - | - | 753.9 | 1114.0 | max_eigenvalue_error=1.044e-06, relative_residual=1.016e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'UPLO': 'L'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### gaussian-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gaussian-nb.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 125.7 | 125.7..125.7 | 1 | - | - | 2931.8 | 1362.0 | accuracy=0.876570, logloss=3.416741 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'var_smoothing': 1e-09}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 45.3 | 45.3..45.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### gaussian-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.gaussian-rp.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 31.4 | 31.4..31.4 | 1 | - | - | 1813.0 | 438.0 | mean_abs_distortion=0.443920 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), GaussianRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| eps | 0.1 |
| n_components | 10 |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 11.6 | 11.6..11.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### gcn / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 13.7 | 13.7..13.7 | 1 | - | - | 1426.9 | 1934.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 4.5 | 4.5..4.5 | 1 | - | - | 1348.5 | 399.4 | max_rel_diff_vs_torch_eager_fp32=0.009646, rel_fro_vs_torch_eager_fp32=1.041e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 13.7 | 13.7..13.7 | 1 | - | - | 1424.0 | 1934.6 | max_rel_diff_vs_torch_eager_fp32=281.122146, rel_fro_vs_torch_eager_fp32=0.0002661 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 8.2 | 8.2..8.2 | 1 | - | - | 1315.1 | 399.4 | max_rel_diff_vs_torch_eager_fp32=281.117866, rel_fro_vs_torch_eager_fp32=0.0002661 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 13.8 | 13.8..13.8 | 1 | - | - | 1548.3 | 2323.8 | max_rel_diff_vs_torch_eager_fp32=2718.059111, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 8.5 | 8.5..8.5 | 1 | - | - | 1571.5 | 782.4 | max_rel_diff_vs_torch_eager_fp32=2718.055850, rel_fro_vs_torch_eager_fp32=0.002187 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'add_self_loops': True, 'bias': True, 'improved': False, 'normalize': True, 'out_channels': 128}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | true | true | true | true | true | true |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 6.9 | 6.9..6.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 6.9 | 6.9..6.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 5.4 | 5.4..5.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-avgpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-avgpool.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 0.7 | 0.7..0.7 | 1 | - | - | 692.6 | 12.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 894.7 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 0.7 | 0.7..0.7 | 1 | - | - | 692.6 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 0.9 | 0.9..0.9 | 1 | - | - | 842.8 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 692.4 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 0.8 | 0.8..0.8 | 1 | - | - | 838.2 | 12.6 | max_rel_diff_vs_torch_eager_fp32=0.009731, rel_fro_vs_torch_eager_fp32=9.64e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'output_size': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.1 | 0.1..0.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### graphsage / istella (rows full, shape X 100000x220; indices 1521510; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 8.6 | 8.6..8.6 | 1 | - | - | 1263.5 | 1667.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 4.2 | 4.2..4.2 | 1 | - | - | 1310.3 | 403.7 | max_rel_diff_vs_torch_eager_fp32=0.134110, rel_fro_vs_torch_eager_fp32=9.7e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 8.5 | 8.5..8.5 | 1 | - | - | 1259.5 | 1667.4 | max_rel_diff_vs_torch_eager_fp32=430.934131, rel_fro_vs_torch_eager_fp32=0.0002969 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 3.8 | 3.8..3.8 | 1 | - | - | 1257.6 | 403.8 | max_rel_diff_vs_torch_eager_fp32=430.934131, rel_fro_vs_torch_eager_fp32=0.0002969 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 8.1 | 8.1..8.1 | 1 | - | - | 1399.9 | 1642.9 | max_rel_diff_vs_torch_eager_fp32=3907.114267, rel_fro_vs_torch_eager_fp32=0.003366 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.3 | 3.3..3.3 | 1 | - | - | 1447.6 | 330.6 | max_rel_diff_vs_torch_eager_fp32=3678.210080, rel_fro_vs_torch_eager_fp32=0.003054 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'aggr': 'mean', 'bias': True, 'normalize': False, 'out_channels': 128, 'project': False, 'root_weight': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| normalize | false | false | false | false | false | false |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 7.7 | 7.7..7.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 7.5 | 7.5..7.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 7.4 | 7.4..7.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### gru-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1103.7 | 1103.7..1103.7 | 1 | - | - | 1122.3 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1190.3 | 1190.3..1190.3 | 1 | - | - | 1174.3 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1063.3 | 1063.3..1063.3 | 1 | - | - | 1123.7 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1248.7 | 1248.7..1248.7 | 1 | - | - | 1174.6 | 643.7 | accuracy=0.971842 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1548.3 | 1548.3..1548.3 | 1 | - | - | 1487.8 | 340.9 | accuracy=0.971951 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1311.6 | 1311.6..1311.6 | 1 | - | - | 1539.4 | 340.9 | accuracy=0.971951 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 2.9 | 2.9..2.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.7 | 2.7..2.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 5.2 | 5.2..5.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.7 | 3.7..3.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1133.5 | 1133.5..1133.5 | 1 | - | - | 1136.9 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1143.5 | 1143.5..1143.5 | 1 | - | - | 1188.0 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1020.0 | 1020.0..1020.0 | 1 | - | - | 1137.7 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1145.8 | 1145.8..1145.8 | 1 | - | - | 1190.4 | 643.4 | finite=True, r2=0.981946, rmse=0.155672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1528.3 | 1528.3..1528.3 | 1 | - | - | 1534.4 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1333.4 | 1333.4..1333.4 | 1 | - | - | 1585.9 | 340.6 | finite=True, r2=0.981898, rmse=0.155878 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 3.0 | 3.0..3.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### incremental-pca / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.incremental-pca.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 1515.2 | 1515.2..1515.2 | 1 | - | - | 3076.8 | 1314.0 | explained_variance_fraction=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'batch_size': 65536, 'n_components': 10, 'whiten': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), IncrementalPCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| batch_size | 65536 |
| n_components | 10 |
| seed | "none (deterministic)" |
| whiten | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### ivf-pq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-pq.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 1905.2 | 1905.2..1905.2 | 1 | - | - | 1793.1 | 822.0 | recall_at_10=0.791300 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuvs-gpu`, seed 7): MATCHED

| parameter | cuvs-gpu |
|---|---|
| library (source) | cuvs (declared) |
| n_neighbors | 10 |
| nlist | 1024 |
| nprobe | 32 |
| seed | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: cuvs-gpu seed: cuVS IndexParams take no seed and the index build samples rows; ours and faiss get 7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | Xq | - | 11.8 | 11.8..11.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### ivf-refine / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-refine.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 1255.0 | 1255.0..1255.0 | 1 | - | - | 1815.7 | 822.0 | recall_at_10=0.993625 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'pq_bits': 8, 'random_state': 7, 'refine_ratio': 4}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuvs-gpu`, seed 7): MATCHED

| parameter | cuvs-gpu |
|---|---|
| library (source) | cuvs (declared) |
| n_neighbors | 10 |
| nlist | 1024 |
| nprobe | 32 |
| seed | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: cuvs-gpu seed: cuVS IndexParams take no seed and the index build samples rows; ours and faiss get 7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | Xq | - | 82.4 | 82.4..82.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### ivf-sq / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/algos.ivf-sq.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 266.7 | 266.7..266.7 | 1 | - | - | 1765.1 | 904.0 | recall_at_10=0.469725 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'kmeans_n_iters': 20, 'n_lists': 1024, 'n_neighbors': 10, 'n_probes': 32, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuvs-gpu`, seed 7): MATCHED

| parameter | cuvs-gpu |
|---|---|
| library (source) | cuvs (declared) |
| n_neighbors | 10 |
| nlist | 1024 |
| nprobe | 32 |
| seed | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: cuvs-gpu seed: cuVS IndexParams take no seed and the index build samples rows; ours and faiss get 7

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | Xq | - | 6.0 | 6.0..6.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### kbins / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.kbins.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 739.0 | 739.0..739.0 | 1 | - | - | 3068.2 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'encode': 'ordinal', 'n_bins': 16, 'quantile_method': 'linear', 'random_state': 7, 'strategy': 'quantile', 'subsample': None}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), KBinsDiscretizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| max_bin | 16 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 26.0 | 26.0..26.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### kernel-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.kernel-shap.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 15329.7 | 15329.7..15329.7 | 1 | - | - | 2042.9 | 438.0 | rel_error_vs_exact=0.032177 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (declared) |
| seed | 7 |

### label-binarizer / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-binarizer.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 18.9 | 18.9..18.9 | 1 | - | - | 1423.1 | 558.0 | output_shape=100000x16 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'neg_label': 0, 'pos_label': 1}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelBinarizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 6.8 | 6.8..6.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### label-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-encoder.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 7.9 | 7.9..7.9 | 1 | - | - | 1022.1 | 490.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), LabelEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 43.4 | 43.4..43.4 | 1 | - | - | 3008.6 | 1374.0 | finite=True, r2=0.328088, rmse=0.684726 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| eps | 2.220446049250313e-16 |
| fit_intercept | true |
| precompute | "auto" |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 6.3 | 6.3..6.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### layernorm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.layernorm.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | - | - | 734.0 | 320.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 7.1 | 7.1..7.1 | 1 | - | - | 984.7 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 734.2 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 935.4 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 734.4 | 320.1 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 930.1 | 336.1 | max_rel_diff_vs_torch_eager_fp32=0.006419, rel_fro_vs_torch_eager_fp32=5.316e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'elementwise_affine': True, 'eps': 1e-05, 'normalized_shape': 1024}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.8 | 0.8..0.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### louvain / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.louvain.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cugraph-gpu | cugraph | gpu | opponent | 83.3 | 83.3..83.3 | 1 | - | - | 917.3 | 436.0 | modularity=0.941795, n_communities=62 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'resolution': 1.0, 'seed': 7}. Rows: None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cugraph-gpu`, seed 7): MATCHED

| parameter | cugraph-gpu |
|---|---|
| library (source) | cugraph (declared) |
| seed | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: cugraph-gpu seed: cuGraph louvain takes no seed; its GPU move order is not seeded; ours and networkx get 7

### lstm-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.taxi-hourly.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1246.5 | 1246.5..1246.5 | 1 | - | - | 1122.8 | 695.4 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1257.4 | 1257.4..1257.4 | 1 | - | - | 1173.7 | 695.4 | accuracy=0.868218 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1230.4 | 1230.4..1230.4 | 1 | - | - | 1124.0 | 695.4 | accuracy=0.868164 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1277.4 | 1277.4..1277.4 | 1 | - | - | 1175.5 | 695.4 | accuracy=0.868164 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1805.5 | 1805.5..1805.5 | 1 | - | - | 1490.6 | 367.2 | accuracy=0.868327 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1735.3 | 1735.3..1735.3 | 1 | - | - | 1542.3 | 367.2 | accuracy=0.868327 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 3.6 | 3.6..3.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 5.7 | 5.7..5.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 6.0 | 6.0..6.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.taxi-hourly.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1330.9 | 1330.9..1330.9 | 1 | - | - | 1131.8 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1243.7 | 1243.7..1243.7 | 1 | - | - | 1183.5 | 695.1 | finite=True, r2=0.751679, rmse=0.540430 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1253.8 | 1253.8..1253.8 | 1 | - | - | 1133.7 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1285.2 | 1285.2..1285.2 | 1 | - | - | 1185.1 | 695.1 | finite=True, r2=0.751680, rmse=0.540429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1711.2 | 1711.2..1711.2 | 1 | - | - | 1536.4 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1539.5 | 1539.5..1539.5 | 1 | - | - | 1587.7 | 366.9 | finite=True, r2=0.751712, rmse=0.540394 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 6.4 | 6.4..6.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 6.7 | 6.7..6.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstsq / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lstsq.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 7.1 | 7.1..7.1 | 1 | - | - | 913.0 | 1122.8 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 9.9 | 9.9..9.9 | 1 | - | - | 727.9 | 2778.0 | relative_residual=0.756366 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lu-solve / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-solve.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 36.1 | 36.1..36.1 | 1 | - | - | 824.1 | 528.3 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 35.8 | 35.8..35.8 | 1 | - | - | 778.4 | 984.0 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### maxabs-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.maxabs-scaler.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 3.4 | 3.4..3.4 | 1 | - | - | 892.7 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MaxAbsScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.9 | 0.9..0.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### maxpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool2d.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | - | - | 719.8 | 639.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 1221.8 | 552.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 720.0 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.5 | 2.5..2.5 | 1 | - | - | 913.7 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.2 | 2.2..2.2 | 1 | - | - | 720.0 | 639.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.8 | 2.8..2.8 | 1 | - | - | 908.0 | 553.5 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'dilation': 1, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### minmax-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minmax-scaler.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 5.1 | 5.1..5.1 | 1 | - | - | 891.9 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'clip': False, 'feature_range': [0, 1]}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params + fixed unclipped MinMaxScaler behavior) |
| clip | false |
| feature_range | [0, 1] |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### multinomial-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multinomial-nb.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 56.7 | 56.7..56.7 | 1 | - | - | 3910.6 | 1372.0 | accuracy=0.853620, logloss=3.628599 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### multinomial-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.multinomial-nb.text.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 50.0 | 50.0..50.0 | 1 | - | - | 4711.5 | 1946.0 | accuracy=0.983067, logloss=0.559524 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 1.0, 'fit_prior': True, 'force_alpha': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), MultinomialNB (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### normalizer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.normalizer.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 2.2 | 2.2..2.2 | 1 | - | - | 3051.7 | 1450.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'norm': 'l2'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), Normalizer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### onehot / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.onehot.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 52.0 | 52.0..52.0 | 1 | - | - | 1251.6 | 524.0 | output_shape=100000x119 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'categories': 'auto', 'handle_unknown': 'ignore', 'sparse_output': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), OneHotEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 54.8 | 54.8..54.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### pagerank / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.pagerank.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cugraph-gpu | cugraph | gpu | opponent | 5.0 | 5.0..5.0 | 1 | - | - | 1009.8 | 440.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cugraph-gpu`, seed 7): MATCHED

| parameter | cugraph-gpu |
|---|---|
| library (source) | cugraph (declared) |
| alpha | 0.85 |
| max_iter | 100 |
| seed | "none (deterministic)" |
| tol | 1e-06 |

### permutation-shap / istella (rows full, shape X 100000x220; Xq 100x220; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.permutation-shap.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 2320.3 | 2320.3..2320.3 | 1 | - | - | 1915.3 | 438.0 | rel_error_vs_exact=1.483e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (declared) |
| seed | 7 |

### poly-features / istella (rows full, shape X 1000000x16; Xq 100000x16; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poly-features.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | 1945.9 | 560.0 | output_shape=100000x152 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'degree': 2, 'include_bias': False, 'interaction_only': False}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PolynomialFeatures (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| degree | 2 |
| order | "C" |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 5.9 | 5.9..5.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### power-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 1, log `logs/algos.power-transformer.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "BracketError('The algorithm terminated without finding a valid bracket. Consider trying different initial points.')", "event": "error", "stage": "round 0"}) (measured this run) |

settings: {'method': 'yeo-johnson', 'standardize': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), PowerTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | - | - | 0 | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "BracketError('The algorithm terminated without finding a valid bracket. Consider trying different initial points.')", "event": "error", "stage": "round 0"}) |

inference call, cuml-gpu: transform(Xq)(Xq)

### qn-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qn-reg.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 232.9 | 232.9..232.9 | 1 | - | - | 2976.8 | 1282.0 | finite=True, r2=0.327567, rmse=0.684991 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'fit_intercept': True, 'l1_strength': 0.0, 'l2_strength': 0.0, 'lbfgs_memory': 5, 'linesearch_max_iter': 50, 'loss': 'squared_error', 'max_iter': 1000, 'penalty_normalized': True, 'tol': 0.0001}. Rows: None. Timed: None.

mismatch: scikit-learn LinearRegression solves the same least-squares problem in closed form (scipy lstsq); it has no max_iter, tol or L-BFGS settings

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| fit_intercept | true |
| loss | "l2" |
| max_iter | 1000 |
| seed | "none (deterministic)" |
| tol | 0.0001 |

### qr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 227.1 | 227.1..227.1 | 1 | - | - | 1703.4 | 3008.9 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 230.5 | 230.5..230.5 | 1 | - | - | 2486.4 | 5160.0 | relative_gram_difference=3.787e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### quantile-transformer / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile-transformer.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 6660.9 | 6660.9..6660.9 | 1 | - | - | 3847.2 | 1444.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'ignore_implicit_zeros': False, 'n_quantiles': 1000, 'output_distribution': 'uniform', 'random_state': 7, 'subsample': 1000000000}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), QuantileTransformer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | 7 |
| subsample | 1000000000 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 282.0 | 282.0..282.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### randomized-svd / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.randomized-svd.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 65.7 | 65.7..65.7 | 1 | - | - | 1637.5 | 1012.2 | relative_reconstruction_error=0.0002359 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### resnet-block / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.resnet-block.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 5.3 | 5.3..5.3 | 1 | - | - | 942.5 | 450.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 5.3 | 5.3..5.3 | 1 | - | - | 1146.2 | 409.2 | max_rel_diff_vs_torch_eager_fp32=0.818182, rel_fro_vs_torch_eager_fp32=5.381e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 4.4 | 4.4..4.4 | 1 | - | - | 959.7 | 554.3 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 4.3 | 4.3..4.3 | 1 | - | - | 1109.0 | 516.8 | max_rel_diff_vs_torch_eager_fp32=1532.793045, rel_fro_vs_torch_eager_fp32=0.0003478 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.8 | 2.8..2.8 | 1 | - | - | 1002.7 | 424.9 | max_rel_diff_vs_torch_eager_fp32=17838.627100, rel_fro_vs_torch_eager_fp32=0.003621 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | - | - | 1173.4 | 383.4 | max_rel_diff_vs_torch_eager_fp32=18497.318029, rel_fro_vs_torch_eager_fp32=0.003425 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'inplanes': 64, 'planes': 64}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 1.0 | 1.0..1.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.9 | 0.9..0.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### rnn-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 904.6 | 904.6..904.6 | 1 | - | - | 1122.2 | 404.2 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1154.2 | 1154.2..1154.2 | 1 | - | - | 1173.6 | 404.2 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 876.8 | 876.8..876.8 | 1 | - | - | 1123.6 | 404.2 | accuracy=0.953505 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1181.2 | 1181.2..1181.2 | 1 | - | - | 1174.6 | 404.2 | accuracy=0.953505 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1223.4 | 1223.4..1223.4 | 1 | - | - | 1459.3 | 220.9 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 873.9 | 873.9..873.9 | 1 | - | - | 1510.4 | 220.9 | accuracy=0.953559 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-reg.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1095.6 | 1095.6..1095.6 | 1 | - | - | 1138.3 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 902.4 | 902.4..902.4 | 1 | - | - | 1188.9 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1136.5 | 1136.5..1136.5 | 1 | - | - | 1133.1 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1161.7 | 1161.7..1161.7 | 1 | - | - | 1184.7 | 403.9 | finite=True, r2=0.977347, rmse=0.174374 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1644.0 | 1644.0..1644.0 | 1 | - | - | 1505.8 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1442.1 | 1442.1..1442.1 | 1 | - | - | 1556.6 | 220.5 | finite=True, r2=0.977345, rmse=0.174385 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'batch_size': 256, 'hidden_size': 64, 'learning_rate': 0.001, 'max_epochs': 2, 'nonlinearity': 'tanh', 'num_layers': 1, 'optimizer': 'adam', 'optimizer_options': {'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}, 'random_state': 7, 'shuffle': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| batch_size | 256 | 256 | 256 | 256 | 256 | 256 |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| hidden_size | 64 | 64 | 64 | 64 | 64 | 64 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| num_layers | 1 | 1 | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| shuffle | true | true | true | true | true | true |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 4.8 | 4.8..4.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 4.4 | 4.4..4.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### robust-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.robust-scaler.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 828.4 | 828.4..828.4 | 1 | - | - | 3064.3 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'quantile_range': [25.0, 75.0], 'unit_variance': False, 'with_centering': True, 'with_scaling': True}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), RobustScaler (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 4.2 | 4.2..4.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 4836.8 | 4836.8..4836.8 | 1 | - | - | 3017.4 | 1374.0 | accuracy=0.809350 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 0.0001 |
| batch_size | 4096 |
| epochs | 100 |
| eta0 | 0.005 |
| fit_intercept | true |
| l1_ratio | 0.15 |
| learning_rate | "constant" |
| loss | "hinge" |
| penalty | "l2" |
| seed | "none (deterministic)" |
| shuffle | true |
| tol | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.8 | 0.8..0.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### sgd-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 4857.9 | 4857.9..4857.9 | 1 | - | - | 2928.5 | 1374.0 | finite=True, r2=0.327768, rmse=0.684889 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 0.0001 |
| batch_size | 4096 |
| epochs | 100 |
| eta0 | 0.005 |
| fit_intercept | true |
| l1_ratio | 0.15 |
| learning_rate | "constant" |
| loss | "squared_loss" |
| penalty | "l2" |
| seed | "none (deterministic)" |
| shuffle | true |
| tol | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### sgd / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.sgd.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 9.6 | 9.6..9.6 | 1 | - | - | 925.9 | 832.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 162.1 | 162.1..162.1 | 1 | - | - | 1175.0 | 832.0 | rel_fro_vs_torch_eager_fp32=6.772e-10 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'dampening': 0.0, 'lr': 0.001, 'maximize': False, 'momentum': 0.9, 'nesterov': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| dampening | 0.0 | 0.0 |
| learning_rate | 0.001 | 0.001 |
| momentum | 0.9 | 0.9 |
| nesterov | false | false |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### simple-imputer / taxi (rows full, shape X 1000000x11; X_true 1000000x11; Xq 100000x11; Xq_true 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.simple-imputer.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 22.4 | 22.4..22.4 | 1 | - | - | 1043.3 | 488.0 | masked_rmse=5.985180 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'add_indicator': False, 'keep_empty_features': False, 'strategy': 'median'}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SimpleImputer (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### sparse-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.sparse-rp.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 5.9 | 5.9..5.9 | 1 | - | - | 932.9 | 430.0 | mean_abs_distortion=0.266849 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 10, 'random_state': 7}. Rows: None. Timed: None.

config: cuML benchmark (RAPIDS), SparseRandomProjection (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| eps | 0.1 |
| n_components | 10 |
| seed | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### standard-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.standard-scaler.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 53.5 | 53.5..53.5 | 1 | - | - | 932.4 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'with_mean': True, 'with_std': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| seed | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### svd / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 8.6 | 8.6..8.6 | 1 | - | - | 850.4 | 1186.8 | max_rel_singular_value_error=2.595e-06, relative_reconstruction_error_100k_rows=7.244e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 8.3 | 8.3..8.3 | 1 | - | - | 658.5 | 2766.0 | max_rel_singular_value_error=3.763e-06, relative_reconstruction_error_100k_rows=6.886e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### svgp / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpytorch-gpu | gpytorch | gpu | opponent | 29.1 | 29.1..29.1 | 1 | - | - | 1184.3 | 1395.0 | finite=True, r2=-0.209527, rmse=17.831018 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| gpytorch-gpu | Xq | - | 7.2 | 7.2..7.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, gpytorch-gpu: predict(Xq)(Xq)

### target-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.target-encoder.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 1170.7 | 1170.7..1170.7 | 1 | - | - | 1331.0 | 614.0 | output_shape=100000x5 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'categories': 'auto', 'cv': 4, 'random_state': 42, 'shuffle': True, 'smooth': 0.0, 'target_type': 'binary'}. Rows: None. Timed: None.

mismatch: fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)

config: cuML benchmark (RAPIDS), TargetEncoder (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 42): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| cv | 4 |
| seed | 42 |
| smooth | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | - | 29.8 | 29.8..29.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### tree-shap / taxi (rows full, shape X 100000x11; Xq 10000x11; y 100000; yq 10000)

race: done, driver rc 0, log `logs/algos.tree-shap.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 51.2 | 51.2..51.2 | 1 | - | - | 485.3 | 500.0 | max_additivity_error=5.402e-05 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}. Rows: None. Timed: None.

mismatch: ours explains its RandomForestRegressor (TreeExplainer takes RF, ExtraTrees, DecisionTree and DART models), the opponents their GBDT of the same size

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `xgboost-gpu`, seed 7): MATCHED

| parameter | xgboost-gpu |
|---|---|
| library (source) | xgboost (declared) |
| learning_rate | 0.1 |
| max_depth | 6 |
| n_estimators | 100 |
| seed | 7 |

### tsne / taxi (rows full, shape X 20000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/algos.tsne.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 396.9 | 396.9..396.9 | 1 | - | - | 918.1 | 448.0 | trustworthiness_k15=0.998353 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'early_exaggeration': 12.0, 'init': 'seeded', 'learning_rate': 'auto', 'max_iter': 1000, 'n_components': 2, 'perplexity': 30.0, 'random_state': 7}. Rows: None. Timed: None.

mismatch: gradients: ours exact repulsion over k-NN affinities (no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, scikit-learn only); cuML FFT

mismatch: init: ours and scikit-learn start from the SAME array, ours' 'random' rule (default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' and draws its own start

mismatch: scikit-learn's early stop is switched off (n_iter_without_progress=1000, min_grad_norm=0.0): ours runs exactly max_iter steps

config: cuML benchmark (RAPIDS), TSNE (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| early_exaggeration | 12.0 |
| init | "random" |
| learning_rate | 200.0 |
| max_iter | 1000 |
| metric | "euclidean" |
| n_components | 2 |
| n_neighbors | 90 |
| perplexity | 30.0 |
| seed | 7 |

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
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)

