# mojolearn benchmark board

Generated 2026-10-06T08:52:48Z from `board.json` (schema `mojolearn-bench-board/1`).

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

Races: 113 planned, 104 done, 3 failed, 0 unsupported, 6 pending. Cells: 236 (REFUSED 9, ok 227).

Inference cells: 156 (REFUSED 6, ok 150).

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
| classical | dbscan | taxi | n_clusters | - | - | cuml-gpu 36 |
| classical | dbscan | taxi | noise_fraction | - | - | cuml-gpu 0.000174 |
| classical | dbscan | taxi | rows | - | - | cuml-gpu 1000000 |
| classical | hdbscan | taxi | n_clusters | - | - | cuml-gpu 159 |
| classical | hdbscan | taxi | noise_fraction | - | - | cuml-gpu 0.130970 |
| classical | hdbscan | taxi | rows | - | - | cuml-gpu 100000 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | - | - | cuml-gpu -14.826437 |
| classical | kde | taxi | rows_without_density | - | - | cuml-gpu 0 |
| classical | kmeans | taxi | inertia (lower is better) | - | - | cuml-gpu 3.06e+08; torch-gpu 3.06e+08 |
| classical | kmeans | taxi | n_iter | - | - | cuml-gpu 32; torch-gpu 58 |
| classical | knn | taxi | recall_at_k (higher is better) | - | - | cuml-gpu 0.999742; torch-gpu 0.999773 |
| classical | knn | taxi | rows_with_repeated_ids | - | - | cuml-gpu 0; torch-gpu 0 |
| classical | ols | taxi | r2 (higher is better) | - | - | cuml-gpu 0.908836; torch-gpu 0.908836; torch-gpu-eigh 0.908836 |
| classical | ols | taxi | rmse (lower is better) | - | - | cuml-gpu 4.696488; torch-gpu 4.696480; torch-gpu-eigh 4.696490 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | - | cuml-gpu 0.999996; torch-gpu 0.999996 |
| classical | svc | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.767500 |
| classical | svc | taxi | n_support | - | - | cuml-gpu 5541 |
| classical2 | agglomerative | taxi | n_clusters | - | - | cuml-gpu 8 |
| classical2 | agglomerative | taxi | silhouette (higher is better) | - | - | cuml-gpu 0.685524 |
| classical2 | elasticnet | istella | r2 (higher is better) | - | - | cuml-gpu 0.260922 |
| classical2 | elasticnet | istella | rmse (lower is better) | - | - | cuml-gpu 0.718134 |
| classical2 | ivf | taxi | recall_at_k (higher is better) | - | - | cuvs-gpu 0.999450 |
| classical2 | ivf | taxi | rows_with_repeated_ids | - | - | cuvs-gpu 0 |
| classical2 | kernel-ridge | taxi | r2 (higher is better) | - | - | cuml-gpu 0.726543 |
| classical2 | kernel-ridge | taxi | rmse (lower is better) | - | - | cuml-gpu 8.330375 |
| classical2 | knn-clf | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.741750 |
| classical2 | knn-reg | taxi | r2 (higher is better) | - | - | cuml-gpu 0.937323 |
| classical2 | knn-reg | taxi | rmse (lower is better) | - | - | cuml-gpu 3.842028 |
| classical2 | lasso | taxi | r2 (higher is better) | - | - | cuml-gpu 0.908995 |
| classical2 | lasso | taxi | rmse (lower is better) | - | - | cuml-gpu 4.804745 |
| classical2 | linearsvc | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.762990 |
| classical2 | linearsvr | taxi | r2 (higher is better) | - | - | cuml-gpu 0.899818 |
| classical2 | linearsvr | taxi | rmse (lower is better) | - | - | cuml-gpu 5.041184 |
| classical2 | logreg | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.763350 |
| classical2 | logreg | taxi | logloss (lower is better) | - | - | cuml-gpu 0.538986 |
| classical2 | logreg | taxi | nonfinite_proba_rows | - | - | cuml-gpu 0 |
| classical2 | ridge | taxi | r2 (higher is better) | - | - | cuml-gpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | - | cuml-gpu 4.805051 |
| classical2 | spectral-embedding | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | - | cuml-gpu 0.891595 |
| classical2 | spectral | taxi | n_clusters | - | - | cuml-gpu 8 |
| classical2 | spectral | taxi | silhouette (higher is better) | - | - | cuml-gpu 0.087609 |
| classical2 | svr | taxi | r2 (higher is better) | - | - | cuml-gpu 0.767551 |
| classical2 | svr | taxi | rmse (lower is better) | - | - | cuml-gpu 7.680405 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | - | - | cuml-gpu 0.999964 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | - | - | cuml-gpu 0.003257 |
| classical2 | umap | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | - | cuml-gpu 0.992305 |
| neural | gemm-int8 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | - | torch-eager-int8 0.000000; torch-compile-int8 0.000000 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | - | torch-eager-fp32 9.018733; torch-eager-tf32 9.018733; torch-compile-fp32 9.018733; torch-compile-tf32 9.018732; torch-eager-bf16 9.018664; torch-compile-bf16 9.018669 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | - | torch-eager-fp32 5.635910; torch-eager-tf32 5.635948; torch-compile-fp32 5.635910; torch-compile-tf32 5.635950; torch-eager-bf16 5.635952; torch-compile-bf16 5.635985 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | - | catboost-gpu 0.156748; xgboost-gpu 0.149263; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | - | catboost-gpu 0.983152; xgboost-gpu 0.983622; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | - | catboost-gpu 0.149188; xgboost-gpu 0.149263; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.356515 |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | - | catboost-gpu 0.983668; xgboost-gpu 0.983622; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.500000 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | - | catboost-gpu 0.258149; xgboost-gpu 0.246803; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.243459 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | - | catboost-gpu 0.907780; xgboost-gpu 0.910140; catboost-cpu -; xgboost-cpu -; lightgbm-cuda 0.911402 |

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
| classical | kmeans | taxi | Xq | - | - | - | - | cuml-gpu 2.8 ms (IDENTICAL/arm -); torch-gpu 0.7 ms (IDENTICAL/arm -) |
| classical | ols | taxi | Xq | - | - | - | - | cuml-gpu 1.6 ms (IDENTICAL/arm -); torch-gpu 0.4 ms (IDENTICAL/arm -); torch-gpu-eigh 0.4 ms (IDENTICAL/arm -) |
| classical | pca | taxi | Xq | - | - | - | - | cuml-gpu 2.0 ms (IDENTICAL/arm -); torch-gpu 0.3 ms (IDENTICAL/arm -) |
| classical | svc | taxi | Xq | - | - | - | - | cuml-gpu 7.1 ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | istella | test | - | - | - | - | catboost-gpu 483.3 ms (IDENTICAL/arm -); xgboost-gpu 96.9 ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | istella | large | - | - | - | - | catboost-gpu 888.0 ms (IDENTICAL/arm -); xgboost-gpu 187.0 ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | istella | test | - | - | - | - | catboost-gpu 470.1 ms (IDENTICAL/arm -); xgboost-gpu 96.8 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | istella | large | - | - | - | - | catboost-gpu 901.4 ms (IDENTICAL/arm -); xgboost-gpu 185.4 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | istella | test | - | - | - | - | catboost-gpu 197.3 ms (IDENTICAL/arm -); xgboost-gpu 507.3 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | istella | large | - | - | - | - | catboost-gpu 339.0 ms (IDENTICAL/arm -); xgboost-gpu 1004.5 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |

## Trees

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 9671.9 | 9671.9..9671.9 | 1 | - | - | 5377.9 | 502.0 | auc=0.983152, logloss=0.156748 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 9340.7 | 9340.7..9340.7 | 1 | - | - | 6478.8 | 502.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-gpu,xgboost-gpu leaves=catboost-gpu:105771,xgboost-gpu:107148 spread=0.0129 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `catboost-gpu`, seed 7): MATCHED

| parameter | catboost-gpu | xgboost-gpu |
|---|---||---|---|
| library (source) | catboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbtree" |
| bootstrap_type | "No" | - |
| feature_border_type | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 |
| feature_fraction_bynode | - | 1.0 |
| grow_policy | "Depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | - |
| leaf_estimation_method | "Newton" | - |
| learning_rate | 0.1 | 0.1 |
| loss | "Logloss" | - |
| max_bin | 255 | 255 |
| max_depth | 8 | 8 |
| max_leaves | 256 | 256 |
| min_child_weight | - | 0.0 |
| min_samples_leaf | 1 | - |
| min_split_gain | - | 0.0 |
| n_estimators | 500 | 500 |
| nan_mode | "Min" | - |
| random_strength | 0.0 | - |
| reg_alpha | - | 0.0 |
| reg_lambda | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | - |
| seed | 7 | 7 |
| subsample | null | 1.0 |

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 500000 | 483.3 | 483.3..483.3 | 1 | - | - | auc=0.983152, auc_matches_fit=True, logloss=0.156748, logloss_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 96.9 | 96.9..96.9 | 1 | - | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 888.0 | 888.0..888.0 | 1 | - | - | - | yes | COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 187.0 | 187.0..187.0 | 1 | - | - | - | yes | COMPARABLE | ok |

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 23630.6 | 23630.6..23630.6 | 1 | - | - | 5379.7 | 496.0 | auc=0.983668, logloss=0.149188 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 13126.2 | 13126.2..13126.2 | 1 | - | - | 6475.4 | 496.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | 4935.5 | 4935.5..4935.5 | 1 | - | - | 4205.8 | 544.0 | auc=0.500000, logloss=0.356515 | yes | UNKNOWN | - | ok (measured this run) |

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=lightgbm-cuda leaves=lightgbm-cuda:1 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `lightgbm-cuda`, seed 7): MATCHED

| parameter | lightgbm-cuda |
|---|---|
| library (source) | lightgbm (get_params) |
| boosting_type | "gbdt" |
| class_weight | null |
| feature_fraction | 1.0 |
| learning_rate | 0.1 |
| max_bin | 255 |
| max_depth | 8 |
| max_leaves | 256 |
| min_child_weight | 0.001 |
| min_samples_leaf | 20 |
| min_split_gain | 0.0 |
| n_estimators | 500 |
| reg_alpha | 0.0 |
| reg_lambda | 1.0 |
| scale_pos_weight | 8.85789592328634 |
| seed | 7 |
| subsample | 1.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 500000 | 470.1 | 470.1..470.1 | 1 | - | - | auc=0.983668, auc_matches_fit=True, logloss=0.149188, logloss_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 96.8 | 96.8..96.8 | 1 | - | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 901.4 | 901.4..901.4 | 1 | - | - | - | yes | COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 185.4 | 185.4..185.4 | 1 | - | - | - | yes | COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | yes | UNKNOWN | REFUSED(GPU-INFERENCE-NOT-SUPPORTED: LightGBM Booster.predict executes on CPU) |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | yes | UNKNOWN | REFUSED(GPU-INFERENCE-NOT-SUPPORTED: LightGBM Booster.predict executes on CPU) |

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

inference call, lightgbm-cuda: Booster.predict (CPU; not executed)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 15195.6 | 15195.6..15195.6 | 1 | - | - | 5364.3 | 558.0 | accuracy=0.907780, mlogloss=0.258149 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 33556.0 | 33556.0..33556.0 | 1 | - | - | 6693.0 | 558.0 | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | 170658.0 | 170658.0..170658.0 | 1 | - | - | 4394.5 | 936.0 | accuracy=0.911402, mlogloss=0.243459 | yes | UNKNOWN | - | ok (measured this run) |

memory, catboost-gpu, xgboost-gpu, lightgbm-cuda: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=lightgbm-cuda leaves=lightgbm-cuda:128000 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `lightgbm-cuda`, seed 7): MATCHED

| parameter | lightgbm-cuda |
|---|---|
| library (source) | lightgbm (get_params) |
| boosting_type | "gbdt" |
| class_weight | null |
| feature_fraction | 1.0 |
| learning_rate | 0.1 |
| max_bin | 255 |
| max_depth | 8 |
| max_leaves | 256 |
| min_child_weight | 0.001 |
| min_samples_leaf | 20 |
| min_split_gain | 0.0 |
| n_estimators | 500 |
| reg_alpha | 0.0 |
| reg_lambda | 1.0 |
| seed | 7 |
| subsample | 1.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 500000 | 197.3 | 197.3..197.3 | 1 | - | - | accuracy=0.907780, accuracy_matches_fit=True, mlogloss=0.258149, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 507.3 | 507.3..507.3 | 1 | - | - | accuracy=0.910140, accuracy_matches_fit=True, mlogloss=0.246803, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 339.0 | 339.0..339.0 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 1004.5 | 1004.5..1004.5 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | yes | UNKNOWN | REFUSED(GPU-INFERENCE-NOT-SUPPORTED: LightGBM Booster.predict executes on CPU) |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | yes | UNKNOWN | REFUSED(GPU-INFERENCE-NOT-SUPPORTED: LightGBM Booster.predict executes on CPU) |

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:28: Model is not one-dimensional, GPU eva), the (rows, n_classes) probability matrix

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the (rows, n_classes) probability matrix

inference call, lightgbm-cuda: Booster.predict (CPU; not executed)

## Classical

### dbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.dbscan.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 436957.2 | 436957.2..436957.2 | 1 | - | - | 841.9 | 474.0 | n_clusters=36, noise_fraction=0.000174, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| algorithm | "brute" |
| eps | 3.0 |
| metric | "euclidean" |
| min_samples | 2 |
| seed | "none (deterministic)" |

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 369.1 | 369.1..369.1 | 1 | - | - | 940.5 | 448.0 | n_clusters=159, noise_fraction=0.130970, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10 (scikit-learn 11: the same core distance, the 10th neighbour besides the point), min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: min_samples: ours and cuML 10, scikit-learn 11. The SAME k-th neighbour: cuML's runner.h:68-80 (ours transcribes it) runs the k-NN at min_samples + 1 including the point itself; scikit-learn's kneighbors(X, min_samples) counts the point itself (its HDBSCAN Notes say so). tools/bench_board_params.py maps both to one canonical value

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| allow_single_cluster | false |
| alpha | 1.0 |
| cluster_selection_epsilon | 0.0 |
| cluster_selection_method | "eom" |
| max_cluster_size | 0 |
| metric | "euclidean" |
| min_cluster_size | 100 |
| min_samples | 10 |
| p | null |
| seed | "none (deterministic)" |

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc 0, log `logs/classical.kde.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 3.8 | 3.8..3.8 | 1 | - | - | 799.7 | 436.0 | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| bandwidth | 1.0 |
| kernel | "gaussian" |
| metric | "euclidean" |
| seed | "none (deterministic)" |

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 306.6 | 306.6..306.6 | 1 | - | - | 1227.4 | 614.0 | inertia=3.06e+08, n_iter=32 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 609.7 | 609.7..609.7 | 1 | - | - | 1251.1 | 438.0 | inertia=3.06e+08, n_iter=58 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cuml (get_params) | torch (declared) |
| init | "k-means++" | "k-means++" |
| max_iter | 300 | 300 |
| metric | - | "euclidean" |
| n_clusters | 8 | 8 |
| n_init | 1 | 1 |
| oversampling_factor | 0.0 | - |
| seed | 7 | 7 |
| tol | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | 500000 | 2.8 | 2.8..2.8 | 1 | - | - | eval_inertia=4.571e+07, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.7 | 0.7..0.7 | 1 | - | - | eval_inertia=4.572e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml KMeans.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 33.3 | 33.3..33.3 | 1 | - | - | 818.1 | 452.0 | recall_at_k=0.999742, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 96.5 | 96.5..96.5 | 1 | - | - | 891.5 | 3174.8 | recall_at_k=0.999773, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cuml (get_params) | torch (declared) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 |
| p | 2 | 2 |
| seed | "none (deterministic)" | 7 |

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 22.1 | 22.1..22.1 | 1 | - | - | 1356.3 | 614.0 | finite=True, r2=0.908836, rmse=4.696488 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 39.0 | 39.0..39.0 | 1 | - | - | 1081.6 | 4649.6 | finite=True, r2=0.908836, rmse=4.696480 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 1098.7 | 376.4 | finite=True, r2=0.908836, rmse=4.696490 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu | torch-gpu | torch-gpu-eigh |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | torch (declared) | torch (declared) |
| algorithm | "eig" | - | - |
| fit_intercept | true | true | true |
| seed | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | 500000 | 1.6 | 1.6..1.6 | 1 | - | - | predict_max_rel_err_own_fp64=8.179e-08, r2_eval=0.908836, rmse_eval=4.696488 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.4 | 0.4..0.4 | 1 | - | - | predict_max_rel_err_own_fp64=1.093e-07, r2_eval=0.908836, rmse_eval=4.696479 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu-eigh | Xq | 500000 | 0.4 | 0.4..0.4 | 1 | - | - | predict_max_rel_err_own_fp64=1.444e-07, r2_eval=0.908836, rmse_eval=4.696490 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml LinearRegression.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 20.5 | 20.5..20.5 | 1 | - | - | 1308.2 | 642.0 | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | 1057.9 | 344.4 | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cuml (get_params) | torch (declared) |
| n_components | 10 | 10 |
| seed | "none (deterministic)" | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" |
| tol | 1e-07 | - |
| whiten | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | 500000 | 2.0 | 2.0..2.0 | 1 | - | - | transform_max_rel_err_own_fp64=1.067e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.3 | 0.3..0.3 | 1 | - | - | transform_max_rel_err_own_fp64=1.072e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml PCA.transform(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc 0, log `logs/classical.svc.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 379.4 | 379.4..379.4 | 1 | - | - | 985.0 | 440.0 | accuracy=0.767500, n_support=5541 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| C | 1.0 |
| class_weight | null |
| coef0 | 0.0 |
| degree | 3 |
| gamma | 0.09090909090909091 |
| kernel | "rbf" |
| max_iter | -1 |
| seed | 7 |
| tol | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | 10000 | 7.1 | 7.1..7.1 | 1 | - | - | accuracy_eval=0.767500 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml SVC.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

## Classical, wave 2

### agglomerative / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.agglomerative.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 62.2 | 62.2..62.2 | 1 | - | - | 813.7 | 430.0 | n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

config: cuML benchmark (RAPIDS), AgglomerativeClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| linkage | "single" |
| metric | "euclidean" |
| n_clusters | 8 |
| seed | "none (deterministic)" |

### elasticnet / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.istella.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 319.5 | 319.5..319.5 | 1 | - | - | 2918.9 | 1374.0 | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 0.1 |
| fit_intercept | true |
| l1_ratio | 0.5 |
| max_iter | 1000 |
| seed | "none (deterministic)" |
| selection | "cyclic" |
| solver | "cd" |
| tol | 0.0001 |

### ets / synthetic (rows full, shape Yfit 64x1440; Yhold 64x48)

race: failed, driver rc 1, log `logs/classical2.ets.synthetic.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | - | - | 0 | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "TypeError('Implicit conversion to a host NumPy array via __array__ is not allowed, To explicitly construct a GPU matrix, consider using .to_cupy()\\nTo explicitly construct a host matrix, c) (measured this run) |

settings: trend additive, seasonal additive, seasonal_periods=24, initialization_method='estimated'; ours and cuML start_periods=2, eps=2.24e-3; statsmodels damped_trend=False, use_boxcox=False. Rows: 64 synthetic hourly series, period 24, 1440 fit points, 48 held out. Timed: construct + fit of every series.

mismatch: initialization: ours 'estimated' (its default, statsmodels' definition), statsmodels 'estimated'; cuML has only its heuristic start (start_periods=2), so its row fits the older initialization

mismatch: cuML returns no in-sample predictions; that quality cell is empty

mismatch: trend: ours and cuML are additive-trend with no parameter; statsmodels trend='additive'. eps is ours' and cuML's only; statsmodels fit() uses its own optimizer

mismatch: seed: no arm has a seed argument

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| eps | 0.00224 |
| seasonal | "additive" |
| seasonal_periods | 24 |
| seed | "none (deterministic)" |
| start_periods | 2 |

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/classical2.ivf.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 297.5 | 297.5..297.5 | 1 | - | - | 933.3 | 450.0 | recall_at_k=0.999450, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuvs-gpu`, seed 7): MATCHED

| parameter | cuvs-gpu |
|---|---|
| library (source) | cuvs (declared) |
| metric | "sqeuclidean" |
| n_neighbors | 10 |
| nlist | 1024 |
| nprobe | 32 |
| seed | "none (no argument; draws random numbers, see exceptions)" |

accepted difference: cuvs-gpu seed: cuVS ivf_flat IndexParams takes no seed and its k-means training samples rows; ours and faiss get 7

### kernel-ridge / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 325.9 | 325.9..325.9 | 1 | - | - | 1048.8 | 474.0 | finite=True, r2=0.726543, rmse=8.330375 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| coef0 | 1.0 |
| degree | 3 |
| gamma | 0.09090909090909091 |
| kernel | "rbf" |
| seed | "none (deterministic)" |

### knn-clf / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 8.8 | 8.8..8.8 | 1 | - | - | 974.1 | 440.0 | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| algorithm | "brute" |
| metric | "euclidean" |
| n_neighbors | 10 |
| p | 2 |
| seed | "none (deterministic)" |
| weights | "uniform" |

### knn-reg / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 5.4 | 5.4..5.4 | 1 | - | - | 847.5 | 440.0 | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| algorithm | "brute" |
| metric | "euclidean" |
| n_neighbors | 10 |
| p | 2 |
| seed | "none (deterministic)" |
| weights | "uniform" |

### lasso / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 10.2 | 10.2..10.2 | 1 | - | - | 968.5 | 498.0 | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 0.01 |
| fit_intercept | true |
| max_iter | 1000 |
| seed | "none (deterministic)" |
| selection | "cyclic" |
| solver | "cd" |
| tol | 0.0001 |

### linearsvc / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 39.7 | 39.7..39.7 | 1 | - | - | 1087.0 | 490.0 | accuracy=0.762990 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| C | 1.0 |
| class_weight | null |
| fit_intercept | true |
| loss | "squared_hinge" |
| max_iter | 1000 |
| penalized_intercept | false |
| penalty | "l2" |
| seed | "none (deterministic)" |
| tol | 0.0001 |

### linearsvr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 21.6 | 21.6..21.6 | 1 | - | - | 996.5 | 498.0 | finite=True, r2=0.899818, rmse=5.041184 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| C | 1.0 |
| epsilon | 0.0 |
| fit_intercept | true |
| loss | "epsilon_insensitive" |
| max_iter | 1000 |
| penalized_intercept | false |
| penalty | "l2" |
| seed | "none (deterministic)" |
| tol | 0.0001 |

### logreg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 9.7 | 9.7..9.7 | 1 | - | - | 1084.8 | 498.0 | accuracy=0.763350, logloss=0.538986, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| C | 1.0 |
| class_weight | null |
| fit_intercept | true |
| l1_ratio | null |
| max_iter | 1000 |
| penalty | "l2" |
| seed | "none (deterministic)" |
| solver | "qn" |
| tol | 0.0001 |

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.ridge.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 15.2 | 15.2..15.2 | 1 | - | - | 1000.4 | 534.0 | finite=True, r2=0.908983, rmse=4.805051 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| alpha | 1.0 |
| fit_intercept | true |
| max_iter | null |
| seed | "none (deterministic)" |
| solver | "eig" |
| tol | 0.0001 |

### spectral-embedding / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 73.0 | 73.0..73.0 | 1 | - | - | 971.2 | 478.0 | trustworthiness_k15=0.891595 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows: the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| affinity | "nearest_neighbors" |
| n_components | 2 |
| n_neighbors | 10 |
| seed | 7 |

### spectral / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.spectral.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 212.8 | 212.8..212.8 | 1 | - | - | 1048.9 | 480.0 | n_clusters=8, silhouette=0.087609 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, random_state=42 (the cuML benchmark's SpectralClustering), assign_labels='kmeans', n_components=8. Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

config: cuML benchmark (RAPIDS), SpectralClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 42): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| affinity | "nearest_neighbors" |
| n_clusters | 8 |
| n_components | 8 |
| n_init | 1 |
| n_neighbors | 10 |
| seed | 42 |

### svr / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.svr.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 42.2 | 42.2..42.2 | 1 | - | - | 911.3 | 442.0 | finite=True, r2=0.767551, rmse=7.680405 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| C | 1.0 |
| coef0 | 0.0 |
| degree | 3 |
| epsilon | 0.1 |
| gamma | 0.09090909090909091 |
| kernel | "rbf" |
| max_iter | -1 |
| seed | "none (deterministic)" |
| tol | 0.001 |

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc 0, log `logs/classical2.tsvd.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 5.9 | 5.9..5.9 | 1 | - | - | 982.5 | 516.0 | explained_variance_ratio_sum=0.999964, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| algorithm | "full" |
| n_components | 10 |
| n_iter | 15 |
| seed | 7 |
| tol | 1e-07 |

### umap / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.umap.taxi.rows-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 249.1 | 249.1..249.1 | 1 | - | - | 988.4 | 588.0 | trustworthiness_k15=0.992305 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows: 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

config: cuML benchmark (RAPIDS), UMAP-Unsupervised (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (get_params) |
| init | "spectral" |
| learning_rate | 1.0 |
| local_connectivity | 1.0 |
| metric | "euclidean" |
| min_dist | 0.1 |
| n_components | 2 |
| n_epochs | 500 |
| n_neighbors | 5 |
| negative_sample_rate | 5 |
| repulsion_strength | 1.0 |
| seed | 7 |
| set_op_mix_ratio | 1.0 |
| spread | 1.0 |

## Neural

### gemm-int8 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-int8.gaussian.shape-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-int8 | torch | gpu | opponent | 45.5 | 45.5..45.5 | 1 | - | - | 811.7 | 160.0 | max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-int8 | torch | gpu | opponent | 44.8 | 44.8..44.8 | 1 | - | - | 1023.4 | 160.0 | max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-int8, torch-compile-int8: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-int8`, seed 7): MATCHED

| parameter | torch-compile-int8 | torch-eager-int8 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-forward.bytes.shape-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 46.5 | 46.5..46.5 | 1 | - | - | 1107.8 | 173.2 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 45.2 | 45.2..45.2 | 1 | - | - | 1105.0 | 173.2 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 45.8 | 45.8..45.8 | 1 | - | - | 1206.6 | 154.7 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 44.1 | 44.1..44.1 | 1 | - | - | 1208.3 | 154.7 | mean_nll=9.018732 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 44.3 | 44.3..44.3 | 1 | - | - | 1233.3 | 192.5 | mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 42.5 | 42.5..42.5 | 1 | - | - | 1378.4 | 192.5 | mean_nll=9.018669 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-forward.gaussian.shape-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 150.9 | 150.9..150.9 | 1 | - | - | 1062.2 | 354.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 152.3 | 152.3..152.3 | 1 | - | - | 1041.6 | 354.3 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 193.2 | 193.2..193.2 | 1 | - | - | 1189.9 | 307.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-eager-tf32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-forward.gaussian.shape-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 84.0 | 84.0..84.0 | 1 | - | - | 1191.2 | 219.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 83.4 | 83.4..83.4 | 1 | - | - | 1186.5 | 219.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 12.4 | 12.4..12.4 | 1 | - | - | 3464.6 | 83.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 16.5 | 16.5..16.5 | 1 | - | - | 3072.8 | 83.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 83.9 | 83.9..83.9 | 1 | - | - | 1313.3 | 212.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 16.3 | 16.3..16.3 | 1 | - | - | 3813.6 | 90.5 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-forward.bytes.shape-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 47.0 | 47.0..47.0 | 1 | - | - | 1266.2 | 137.4 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 46.6 | 46.6..46.6 | 1 | - | - | 1245.0 | 137.4 | mean_nll=5.635948 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 12.1 | 12.1..12.1 | 1 | - | - | 1870.8 | 64.2 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 4.9 | 4.9..4.9 | 1 | - | - | 1726.6 | 64.2 | mean_nll=5.635950 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 47.7 | 47.7..47.7 | 1 | - | - | 1371.7 | 142.8 | mean_nll=5.635952 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 5.5 | 5.5..5.5 | 1 | - | - | 2114.4 | 74.1 | mean_nll=5.635985 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-forward.gaussian.shape-full.log`, ran on cc560ebdaf91

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 3.6 | 3.6..3.6 | 1 | - | - | 1099.1 | 78.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 3.6 | 3.6..3.6 | 1 | - | - | 1094.7 | 78.4 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 4.5 | 4.5..4.5 | 1 | - | - | 993.5 | 46.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 4.3 | 4.3..4.3 | 1 | - | - | 950.5 | 46.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | - | - | 1291.8 | 80.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.8 | 3.8..3.8 | 1 | - | - | 1230.9 | 41.8 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

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

