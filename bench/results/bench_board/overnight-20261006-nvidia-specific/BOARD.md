# mojolearn benchmark board

Generated 2026-10-06T05:50:41Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | nvidia / cuda |
| GPU | NVIDIA L40S |
| GPU driver | 580.159.03 |
| CPU | AMD EPYC 9554 64-Core Processor (128 logical cores) |
| memory bytes | 1081850585088 |
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

Races: 114 planned, 101 done, 9 failed, 0 unsupported, 4 pending. Cells: 249 (REFUSED 19, UNKNOWN 18, ok 212).

Inference cells: 164 (REFUSED 2, UNKNOWN 8, ok 154).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | opponents |
|---|---|---|---|---|---|---|
| algos | adafactor | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | adam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 4.891e-08 |
| algos | adamw | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 2.203e-07 |
| algos | als | text | recall_at_10 (higher is better) | - | - | implicit-gpu 0.546926 |
| algos | autoarima | taxi-hourly | forecast_rmse (lower is better) | - | - | cuml-gpu 106.817545 |
| algos | avgpool2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | avgpool2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | batchnorm2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.001863; torch-eager-tf32 0.000000; torch-compile-tf32 0.001863; torch-eager-bf16 0.000000; torch-compile-bf16 0.001863 |
| algos | batchnorm2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 4.327e-08; torch-eager-tf32 0.000000; torch-compile-tf32 4.327e-08; torch-eager-bf16 0.000000; torch-compile-bf16 4.327e-08 |
| algos | bernoulli-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.755560 |
| algos | bernoulli-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 0.557803 |
| algos | cagra | taxi | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.619300 |
| algos | categorical-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.765850 |
| algos | categorical-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 0.538866 |
| algos | complement-nb | istella | accuracy (higher is better) | - | - | cuml-gpu 0.849350 |
| algos | complement-nb | istella | logloss (lower is better) | - | - | cuml-gpu 3.763060 |
| algos | complement-nb | text | accuracy (higher is better) | - | - | cuml-gpu 0.983067 |
| algos | complement-nb | text | logloss (lower is better) | - | - | cuml-gpu 0.559490 |
| algos | connected-components | taxi | n_components | - | - | cugraph-gpu 588 |
| algos | conv2d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.506768; torch-eager-tf32 382.931903; torch-compile-tf32 382.931903; torch-eager-bf16 3418.337554; torch-compile-bf16 3433.596343 |
| algos | conv2d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 5.007e-07; torch-eager-tf32 0.0003021; torch-compile-tf32 0.0003021; torch-eager-bf16 0.003382; torch-compile-bf16 0.003380 |
| algos | dart-reg | istella | r2 (higher is better) | - | - | xgboost-gpu 0.577383 |
| algos | dart-reg | istella | rmse (lower is better) | - | - | xgboost-gpu 0.543042 |
| algos | dart | istella | accuracy (higher is better) | - | - | xgboost-gpu 0.954280 |
| algos | dart | istella | logloss (lower is better) | - | - | xgboost-gpu 0.115097 |
| algos | decision-tree-clf | istella | accuracy (higher is better) | - | - | cuml-gpu 0.935000 |
| algos | decision-tree-clf | istella | logloss (lower is better) | - | - | cuml-gpu 0.758303 |
| algos | decision-tree-reg | istella | r2 (higher is better) | - | - | cuml-gpu 0.379438 |
| algos | decision-tree-reg | istella | rmse (lower is better) | - | - | cuml-gpu 0.658041 |
| algos | embedding | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | embedding | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | gaussian-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.719810 |
| algos | gaussian-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 1.132317 |
| algos | gaussian-rp | taxi | mean_abs_distortion | - | - | cuml-gpu 0.302259 |
| algos | gcn | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.002327; torch-eager-tf32 0.001863; torch-compile-tf32 0.002794; torch-eager-bf16 1509.509282; torch-compile-bf16 1509.508234 |
| algos | gcn | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 9.803e-08; torch-eager-tf32 7.223e-08; torch-compile-tf32 9.801e-08; torch-eager-bf16 0.002330; torch-compile-bf16 0.002330 |
| algos | global-maxpool | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | global-maxpool | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | graphsage | taxi | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.238419; torch-eager-tf32 0.029851; torch-compile-tf32 1475.334167; torch-eager-bf16 5460.333333; torch-compile-bf16 4608.154297 |
| algos | graphsage | taxi | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 6.954e-08; torch-eager-tf32 4.218e-08; torch-compile-tf32 0.0002145; torch-eager-bf16 0.003557; torch-compile-bf16 0.003285 |
| algos | gru-clf | taxi-hourly | accuracy (higher is better) | - | - | torch-eager-fp32 0.865668; torch-compile-fp32 0.865668; torch-eager-tf32 0.865668; torch-compile-tf32 0.865668; torch-eager-bf16 0.865668; torch-compile-bf16 0.865668 |
| algos | gru-reg | taxi-hourly | r2 (higher is better) | - | - | torch-eager-fp32 0.748219; torch-compile-fp32 0.748219; torch-eager-tf32 0.748219; torch-compile-tf32 0.748219; torch-eager-bf16 0.748325; torch-compile-bf16 0.748325 |
| algos | gru-reg | taxi-hourly | rmse (lower is better) | - | - | torch-eager-fp32 0.544182; torch-compile-fp32 0.544182; torch-eager-tf32 0.544182; torch-compile-tf32 0.544182; torch-eager-bf16 0.544067; torch-compile-bf16 0.544067 |
| algos | incremental-pca | taxi | explained_variance_fraction | - | - | cuml-gpu 0.999995 |
| algos | ivf-pq | taxi | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.976875 |
| algos | ivf-refine | taxi | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.999050 |
| algos | ivf-sq | taxi | recall_at_10 (higher is better) | - | - | cuvs-gpu 0.772500 |
| algos | kernel-shap | taxi | rel_error_vs_exact | - | - | cuml-gpu 7.647e-07 |
| algos | lars | taxi | r2 (higher is better) | - | - | cuml-gpu 0.908983 |
| algos | lars | taxi | rmse (lower is better) | - | - | cuml-gpu 4.805052 |
| algos | louvain | istella | modularity | - | - | cugraph-gpu 0.909430 |
| algos | louvain | istella | n_communities | - | - | cugraph-gpu 41 |
| algos | lstm-clf | synthetic | accuracy (higher is better) | - | - | torch-eager-fp32 0.968696; torch-compile-fp32 0.968696; torch-eager-tf32 0.968696; torch-compile-tf32 0.968696; torch-eager-bf16 0.968913; torch-compile-bf16 0.968913 |
| algos | lstm-reg | synthetic | r2 (higher is better) | - | - | torch-eager-fp32 0.981013; torch-compile-fp32 0.981013; torch-eager-tf32 0.981013; torch-compile-tf32 0.981013; torch-eager-bf16 0.980998; torch-compile-bf16 0.980998 |
| algos | lstm-reg | synthetic | rmse (lower is better) | - | - | torch-eager-fp32 0.159641; torch-compile-fp32 0.159641; torch-eager-tf32 0.159642; torch-compile-tf32 0.159642; torch-eager-bf16 0.159706; torch-compile-bf16 0.159706 |
| algos | lstsq | istella | relative_residual | - | - | torch-gpu nan; cupy-gpu 0.849957 |
| algos | lu-factor | synthetic | relative_residual | - | - | torch-gpu 3.386e-07; cupy-gpu 3.386e-07 |
| algos | maxpool1d | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | maxpool1d | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000; torch-eager-tf32 0.000000; torch-compile-tf32 0.000000; torch-eager-bf16 0.000000; torch-compile-bf16 0.000000 |
| algos | moe | synthetic | max_rel_diff_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.026193; torch-eager-tf32 1004.691291; torch-compile-tf32 1004.691291; torch-eager-bf16 22834.612745; torch-compile-bf16 22851.987745 |
| algos | moe | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.548e-07; torch-eager-tf32 0.017534; torch-compile-tf32 0.017534; torch-eager-bf16 0.055222; torch-compile-bf16 0.055199 |
| algos | multinomial-nb | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.723160 |
| algos | multinomial-nb | taxi | logloss (lower is better) | - | - | cuml-gpu 0.590750 |
| algos | nadam | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 1.498e-07 |
| algos | pagerank | taxi | sum | - | - | cugraph-gpu 1.000000 |
| algos | permutation-shap | taxi | rel_error_vs_exact | - | - | cuml-gpu 1.701e-07 |
| algos | qn-reg | taxi | r2 (higher is better) | - | - | cuml-gpu 0.908983 |
| algos | qn-reg | taxi | rmse (lower is better) | - | - | cuml-gpu 4.805040 |
| algos | qr | taxi | relative_gram_difference | - | - | torch-gpu 5.971e-06; cupy-gpu 5.971e-06 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | - | - | torch-gpu 0.027197 |
| algos | rmsprop | synthetic | rel_fro_vs_torch_eager_fp32 | - | - | torch-eager-fp32 -; torch-compile-fp32 0.000000 |
| algos | rnn-clf | taxi-hourly | accuracy (higher is better) | - | - | torch-eager-fp32 0.868056; torch-compile-fp32 0.868056; torch-eager-tf32 0.868056; torch-compile-tf32 0.868056; torch-eager-bf16 0.868001; torch-compile-bf16 0.868001 |
| algos | rnn-reg | taxi-hourly | r2 (higher is better) | - | - | torch-eager-fp32 0.738796; torch-compile-fp32 0.738796; torch-eager-tf32 0.738800; torch-compile-tf32 0.738800; torch-eager-bf16 0.738906; torch-compile-bf16 0.738906 |
| algos | rnn-reg | taxi-hourly | rmse (lower is better) | - | - | torch-eager-fp32 0.554271; torch-compile-fp32 0.554271; torch-eager-tf32 0.554267; torch-compile-tf32 0.554267; torch-eager-bf16 0.554155; torch-compile-bf16 0.554155 |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | - | cuml-gpu 0.703120 |
| algos | sgd-reg | taxi | r2 (higher is better) | - | - | cuml-gpu 0.908979 |
| algos | sgd-reg | taxi | rmse (lower is better) | - | - | cuml-gpu 4.805168 |
| algos | simple-imputer | istella | masked_rmse | - | - | cuml-gpu 346849.129968 |
| algos | sparse-rp | istella | mean_abs_distortion | - | - | cuml-gpu 0.876709 |
| algos | svd | istella | max_rel_singular_value_error | - | - | torch-gpu 8.068917; cupy-gpu 10270.720886 |
| algos | svd | istella | relative_reconstruction_error_100k_rows | - | - | torch-gpu 2.861e-05; cupy-gpu 2.385e-06 |
| algos | svgp | istella | r2 (higher is better) | - | - | gpytorch-gpu -0.106040 |
| algos | svgp | istella | rmse (lower is better) | - | - | gpytorch-gpu 0.878383 |
| algos | tree-shap | istella | max_additivity_error | - | - | xgboost-gpu 1.956e-06 |
| algos | tsne | istella | trustworthiness_k15 (higher is better, 1 at most) | - | - | cuml-gpu 0.990033 |
| classical | dbscan | istella | n_clusters | - | - | cuml-gpu 40131 |
| classical | dbscan | istella | noise_fraction | - | - | cuml-gpu 0.219391 |
| classical | dbscan | istella | rows | - | - | cuml-gpu 1000000 |
| classical | hdbscan | istella | n_clusters | - | - | cuml-gpu 53 |
| classical | hdbscan | istella | noise_fraction | - | - | cuml-gpu 0.256280 |
| classical | hdbscan | istella | rows | - | - | cuml-gpu 100000 |
| classical | kde | istella | mean_log_likelihood (higher is better) | - | - | cuml-gpu -222.270582 |
| classical | kde | istella | rows_without_density | - | - | cuml-gpu 0 |
| classical | kmeans | istella | inertia (lower is better) | - | - | cuml-gpu 6.111e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | n_iter | - | - | cuml-gpu 21; torch-gpu 55 |
| classical | knn | istella | recall_at_k (higher is better) | - | - | cuml-gpu 0.976402; torch-gpu 0.981012 |
| classical | knn | istella | rows_with_repeated_ids | - | - | cuml-gpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | - | cuml-gpu -11031.855105; torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | - | cuml-gpu 87.647429; torch-gpu nan; torch-gpu-eigh 0.768590 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | - | cuml-gpu 1.000000; torch-gpu 1.000000 |
| classical | svc | istella | accuracy (higher is better) | - | - | cuml-gpu 0.922200 |
| classical | svc | istella | n_support | - | - | cuml-gpu 2401 |
| classical2 | agglomerative | istella | n_clusters | - | - | cuml-gpu 8 |
| classical2 | agglomerative | istella | silhouette (higher is better) | - | - | cuml-gpu 0.716728 |
| classical2 | arima | synthetic | forecast_rmse (lower is better) | - | - | cuml-gpu 1.515427 |
| classical2 | arima | synthetic | insample_rmse (lower is better) | - | - | cuml-gpu 0.999338 |
| classical2 | arima | synthetic | mean_aic (lower is better) | - | - | cuml-gpu 5680.957154 |
| classical2 | arima | synthetic | mean_llf (higher is better) | - | - | cuml-gpu -2836.478577 |
| classical2 | elasticnet | taxi | r2 (higher is better) | - | - | cuml-gpu 0.907378 |
| classical2 | elasticnet | taxi | rmse (lower is better) | - | - | cuml-gpu 4.847224 |
| classical2 | ivf | istella | recall_at_k (higher is better) | - | - | cuvs-gpu 0.999975 |
| classical2 | ivf | istella | rows_with_repeated_ids | - | - | cuvs-gpu 0 |
| classical2 | kernel-ridge | istella | r2 (higher is better) | - | - | cuml-gpu 0.385427 |
| classical2 | kernel-ridge | istella | rmse (lower is better) | - | - | cuml-gpu 0.646407 |
| classical2 | knn-clf | istella | accuracy (higher is better) | - | - | cuml-gpu 0.926250 |
| classical2 | knn-reg | istella | r2 (higher is better) | - | - | cuml-gpu 0.418145 |
| classical2 | knn-reg | istella | rmse (lower is better) | - | - | cuml-gpu 0.625388 |
| classical2 | lasso | istella | r2 (higher is better) | - | - | cuml-gpu 0.310837 |
| classical2 | lasso | istella | rmse (lower is better) | - | - | cuml-gpu 0.693460 |
| classical2 | linearsvc | istella | accuracy (higher is better) | - | - | cuml-gpu 0.922860 |
| classical2 | linearsvr | istella | r2 (higher is better) | - | - | cuml-gpu -0.106761 |
| classical2 | linearsvr | istella | rmse (lower is better) | - | - | cuml-gpu 0.878795 |
| classical2 | logreg | istella | accuracy (higher is better) | - | - | cuml-gpu 0.924430 |
| classical2 | logreg | istella | logloss (lower is better) | - | - | cuml-gpu 0.181268 |
| classical2 | logreg | istella | nonfinite_proba_rows | - | - | cuml-gpu 0 |
| classical2 | ridge | istella | r2 (higher is better) | - | - | cuml-gpu -0.251259 |
| classical2 | ridge | istella | rmse (lower is better) | - | - | cuml-gpu 0.934403 |
| classical2 | spectral-embedding | istella | trustworthiness_k15 (higher is better, 1 at most) | - | - | cuml-gpu 0.882904 |
| classical2 | spectral | istella | n_clusters | - | - | cuml-gpu 8 |
| classical2 | spectral | istella | silhouette (higher is better) | - | - | cuml-gpu 0.147757 |
| classical2 | svr | istella | r2 (higher is better) | - | - | cuml-gpu 0.318232 |
| classical2 | svr | istella | rmse (lower is better) | - | - | cuml-gpu 0.680829 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | - | - | cuml-gpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | - | - | cuml-gpu 0.0001472 |
| classical2 | umap | istella | trustworthiness_k15 (higher is better, 1 at most) | - | - | cuml-gpu 0.979912 |
| neural | gemm-bf16 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | - | torch-eager-bf16 0.002764; torch-compile-bf16 0.002764 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | - | torch-eager-fp32 1.401e-06; torch-eager-tf32 0.0002784; torch-compile-fp32 1.401e-06; torch-compile-tf32 0.0002784; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | - | torch-eager-fp32 1.160401; torch-eager-tf32 1.160392; torch-compile-fp32 1.160401; torch-compile-tf32 1.160392; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | - | torch-eager-fp32 1.123361; torch-eager-tf32 1.123355; torch-compile-fp32 1.123361; torch-compile-tf32 1.123355; torch-eager-bf16 1.123461; torch-compile-bf16 1.123462 |
| neural | mlp-train-step | gaussian | steps | - | - | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | - | catboost-gpu 0.630387; xgboost-gpu 0.631978; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | - | catboost-gpu 0.528535; xgboost-gpu 0.528548; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | - | catboost-gpu 0.632335; xgboost-gpu 0.630969; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | - | catboost-gpu 0.527912; xgboost-gpu 0.528678; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | - | catboost-gpu 0.631766; xgboost-gpu 0.630969; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | - | catboost-gpu 0.528065; xgboost-gpu 0.528678; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | - | catboost-gpu 0.599270; xgboost-gpu 0.601200; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | - | catboost-gpu 1.012704; xgboost-gpu 1.005204; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | - | catboost-gpu 0.628945; catboost-cpu - |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | - | catboost-gpu 0.528997; catboost-cpu - |
| trees | gbdt-rank-yetirank | istella | map (higher is better) | - | - | catboost-gpu 0.814091; xgboost-gpu 0.842476; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-rank-yetirank | istella | ndcg10 (higher is better) | - | - | catboost-gpu 0.681202; xgboost-gpu 0.725584; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-rank-yetirank | istella | ndcg5 (higher is better) | - | - | catboost-gpu 0.614769; xgboost-gpu 0.663588; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows. Our CPU is never raced or reported.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|
| algos | avgpool2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.5 ms (IDENTICAL/arm -); torch-compile-fp32 0.6 ms (IDENTICAL/arm -); torch-eager-tf32 0.5 ms (IDENTICAL/arm -); torch-compile-tf32 0.6 ms (IDENTICAL/arm -); torch-eager-bf16 0.5 ms (IDENTICAL/arm -); torch-compile-bf16 0.6 ms (IDENTICAL/arm -) |
| algos | batchnorm2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-tf32 0.2 ms (IDENTICAL/arm -); torch-compile-tf32 0.2 ms (IDENTICAL/arm -); torch-eager-bf16 0.2 ms (IDENTICAL/arm -); torch-compile-bf16 0.2 ms (IDENTICAL/arm -) |
| algos | bernoulli-nb | taxi | Xq | - | - | - | - | cuml-gpu 2.0 ms (IDENTICAL/arm -) |
| algos | binarizer | taxi | Xq | - | - | - | - | cuml-gpu 11.4 ms (IDENTICAL/arm -) |
| algos | cagra | taxi | Xq | - | - | - | - | cuvs-gpu 1275.0 ms (IDENTICAL/arm -) |
| algos | categorical-nb | taxi | Xq | - | - | - | - | cuml-gpu 6.5 ms (IDENTICAL/arm -) |
| algos | complement-nb | istella | Xq | - | - | - | - | cuml-gpu 2.2 ms (IDENTICAL/arm -) |
| algos | complement-nb | text | Xq | - | - | - | - | cuml-gpu 2.4 ms (IDENTICAL/arm -) |
| algos | conv2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.5 ms (IDENTICAL/arm -); torch-compile-fp32 1.2 ms (IDENTICAL/arm -); torch-eager-tf32 0.4 ms (IDENTICAL/arm -); torch-compile-tf32 0.6 ms (IDENTICAL/arm -); torch-eager-bf16 0.3 ms (IDENTICAL/arm -); torch-compile-bf16 0.5 ms (IDENTICAL/arm -) |
| algos | dart-reg | istella | Xq | - | - | - | - | xgboost-gpu 146.3 ms (IDENTICAL/arm -) |
| algos | dart | istella | Xq | - | - | - | - | xgboost-gpu 157.1 ms (IDENTICAL/arm -) |
| algos | decision-tree-clf | istella | Xq | - | - | - | - | cuml-gpu 3.9 ms (IDENTICAL/arm -) |
| algos | decision-tree-reg | istella | Xq | - | - | - | - | cuml-gpu 0.7 ms (IDENTICAL/arm -) |
| algos | dropout2d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.2 ms (IDENTICAL/arm -); torch-compile-fp32 0.3 ms (IDENTICAL/arm -); torch-eager-tf32 0.2 ms (IDENTICAL/arm -); torch-compile-tf32 0.5 ms (IDENTICAL/arm -) |
| algos | embedding | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.4 ms (IDENTICAL/arm -); torch-compile-fp32 0.4 ms (IDENTICAL/arm -) |
| algos | gaussian-nb | taxi | Xq | - | - | - | - | cuml-gpu 8.5 ms (IDENTICAL/arm -) |
| algos | gaussian-rp | taxi | Xq | - | - | - | - | cuml-gpu 3.9 ms (IDENTICAL/arm -) |
| algos | gcn | taxi | Xq | - | - | - | - | torch-eager-fp32 5.5 ms (IDENTICAL/arm -); torch-compile-fp32 1.5 ms (IDENTICAL/arm -); torch-eager-tf32 5.5 ms (IDENTICAL/arm -); torch-compile-tf32 1.5 ms (IDENTICAL/arm -); torch-eager-bf16 4.5 ms (IDENTICAL/arm -); torch-compile-bf16 1.4 ms (IDENTICAL/arm -) |
| algos | global-maxpool | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.1 ms (IDENTICAL/arm -); torch-compile-fp32 0.2 ms (IDENTICAL/arm -); torch-eager-tf32 0.1 ms (IDENTICAL/arm -); torch-compile-tf32 0.1 ms (IDENTICAL/arm -); torch-eager-bf16 0.1 ms (IDENTICAL/arm -); torch-compile-bf16 0.1 ms (IDENTICAL/arm -) |
| algos | graphsage | taxi | Xq | - | - | - | - | torch-eager-fp32 0.8 ms (IDENTICAL/arm -); torch-compile-fp32 0.6 ms (IDENTICAL/arm -); torch-eager-tf32 0.7 ms (IDENTICAL/arm -); torch-compile-tf32 0.4 ms (IDENTICAL/arm -); torch-eager-bf16 0.4 ms (IDENTICAL/arm -); torch-compile-bf16 0.4 ms (IDENTICAL/arm -) |
| algos | gru-clf | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 2.5 ms (IDENTICAL/arm -); torch-compile-fp32 2.5 ms (IDENTICAL/arm -); torch-eager-tf32 2.6 ms (IDENTICAL/arm -); torch-compile-tf32 2.6 ms (IDENTICAL/arm -); torch-eager-bf16 2.1 ms (IDENTICAL/arm -); torch-compile-bf16 2.3 ms (IDENTICAL/arm -) |
| algos | gru-reg | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 2.6 ms (IDENTICAL/arm -); torch-compile-fp32 2.6 ms (IDENTICAL/arm -); torch-eager-tf32 2.6 ms (IDENTICAL/arm -); torch-compile-tf32 2.5 ms (IDENTICAL/arm -); torch-eager-bf16 2.0 ms (IDENTICAL/arm -); torch-compile-bf16 2.2 ms (IDENTICAL/arm -) |
| algos | incremental-pca | taxi | Xq | - | - | - | - | cuml-gpu 9.1 ms (IDENTICAL/arm -) |
| algos | ivf-pq | taxi | Xq | - | - | - | - | cuvs-gpu 4.0 ms (IDENTICAL/arm -) |
| algos | ivf-refine | taxi | Xq | - | - | - | - | cuvs-gpu 57.5 ms (IDENTICAL/arm -) |
| algos | ivf-sq | taxi | Xq | - | - | - | - | cuvs-gpu 1.3 ms (IDENTICAL/arm -) |
| algos | kbins | taxi | Xq | - | - | - | - | cuml-gpu 2.4 ms (IDENTICAL/arm -) |
| algos | label-binarizer | taxi | Xq | - | - | - | - | cuml-gpu 30.1 ms (IDENTICAL/arm -) |
| algos | label-encoder | taxi | Xq | - | - | - | - | cuml-gpu 4.1 ms (IDENTICAL/arm -) |
| algos | lars | taxi | Xq | - | - | - | - | cuml-gpu 0.6 ms (IDENTICAL/arm -) |
| algos | lstm-clf | synthetic | Xq | - | - | - | - | torch-eager-fp32 2.8 ms (IDENTICAL/arm -); torch-compile-fp32 2.8 ms (IDENTICAL/arm -); torch-eager-tf32 2.8 ms (IDENTICAL/arm -); torch-compile-tf32 2.8 ms (IDENTICAL/arm -); torch-eager-bf16 2.3 ms (IDENTICAL/arm -); torch-compile-bf16 2.5 ms (IDENTICAL/arm -) |
| algos | lstm-reg | synthetic | Xq | - | - | - | - | torch-eager-fp32 2.8 ms (IDENTICAL/arm -); torch-compile-fp32 2.8 ms (IDENTICAL/arm -); torch-eager-tf32 2.8 ms (IDENTICAL/arm -); torch-compile-tf32 3.3 ms (IDENTICAL/arm -); torch-eager-bf16 4.2 ms (IDENTICAL/arm -); torch-compile-bf16 2.8 ms (IDENTICAL/arm -) |
| algos | maxabs-scaler | istella | Xq | - | - | - | - | cuml-gpu 6.1 ms (IDENTICAL/arm -) |
| algos | maxpool1d | synthetic | Xq | - | - | - | - | torch-eager-fp32 0.3 ms (IDENTICAL/arm -); torch-compile-fp32 0.3 ms (IDENTICAL/arm -); torch-eager-tf32 0.3 ms (IDENTICAL/arm -); torch-compile-tf32 0.3 ms (IDENTICAL/arm -); torch-eager-bf16 0.3 ms (IDENTICAL/arm -); torch-compile-bf16 0.3 ms (IDENTICAL/arm -) |
| algos | minmax-scaler | istella | Xq | - | - | - | - | cuml-gpu 2.7 ms (IDENTICAL/arm -) |
| algos | moe | synthetic | Xq | - | - | - | - | torch-eager-fp32 11.2 ms (IDENTICAL/arm -); torch-compile-fp32 12.5 ms (IDENTICAL/arm -); torch-eager-tf32 4.4 ms (IDENTICAL/arm -); torch-compile-tf32 4.5 ms (IDENTICAL/arm -); torch-eager-bf16 3.4 ms (IDENTICAL/arm -); torch-compile-bf16 3.7 ms (IDENTICAL/arm -) |
| algos | multinomial-nb | taxi | Xq | - | - | - | - | cuml-gpu 3.5 ms (IDENTICAL/arm -) |
| algos | normalizer | taxi | Xq | - | - | - | - | cuml-gpu 1.4 ms (IDENTICAL/arm -) |
| algos | onehot | taxi | Xq | - | - | - | - | cuml-gpu 44.8 ms (IDENTICAL/arm -) |
| algos | poly-features | taxi | Xq | - | - | - | - | cuml-gpu 4.6 ms (IDENTICAL/arm -) |
| algos | power-transformer | taxi | Xq | - | - | - | - | cuml-gpu 13.7 ms (IDENTICAL/arm -) |
| algos | quantile-transformer | taxi | Xq | - | - | - | - | cuml-gpu 122.2 ms (IDENTICAL/arm -) |
| algos | rnn-clf | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 1.1 ms (IDENTICAL/arm -); torch-compile-fp32 1.3 ms (IDENTICAL/arm -); torch-eager-tf32 1.2 ms (IDENTICAL/arm -); torch-compile-tf32 1.3 ms (IDENTICAL/arm -); torch-eager-bf16 1.8 ms (IDENTICAL/arm -); torch-compile-bf16 1.9 ms (IDENTICAL/arm -) |
| algos | rnn-reg | taxi-hourly | Xq | - | - | - | - | torch-eager-fp32 1.1 ms (IDENTICAL/arm -); torch-compile-fp32 1.3 ms (IDENTICAL/arm -); torch-eager-tf32 2.0 ms (IDENTICAL/arm -); torch-compile-tf32 1.3 ms (IDENTICAL/arm -); torch-eager-bf16 1.8 ms (IDENTICAL/arm -); torch-compile-bf16 3.1 ms (IDENTICAL/arm -) |
| algos | robust-scaler | taxi | Xq | - | - | - | - | cuml-gpu 0.7 ms (IDENTICAL/arm -) |
| algos | sgd-clf | taxi | Xq | - | - | - | - | cuml-gpu 0.9 ms (IDENTICAL/arm -) |
| algos | sgd-reg | taxi | Xq | - | - | - | - | cuml-gpu 0.7 ms (IDENTICAL/arm -) |
| algos | simple-imputer | istella | Xq | - | - | - | - | cuml-gpu 38.2 ms (IDENTICAL/arm -) |
| algos | sparse-rp | istella | Xq | - | - | - | - | cuml-gpu 19.6 ms (IDENTICAL/arm -) |
| algos | standard-scaler | istella | Xq | - | - | - | - | cuml-gpu 1.8 ms (IDENTICAL/arm -) |
| algos | svgp | istella | Xq | - | - | - | - | gpytorch-gpu 9.2 ms (IDENTICAL/arm -) |
| algos | target-encoder | istella | Xq | - | - | - | - | cuml-gpu 53.4 ms (IDENTICAL/arm -) |
| classical | kmeans | istella | Xq | - | - | - | - | cuml-gpu 5.4 ms (IDENTICAL/arm -); torch-gpu 1.2 ms (IDENTICAL/arm -) |
| classical | ols | istella | Xq | - | - | - | - | cuml-gpu 2.3 ms (IDENTICAL/arm -); torch-gpu 0.9 ms (IDENTICAL/arm -); torch-gpu-eigh 0.9 ms (IDENTICAL/arm -) |
| classical | pca | istella | Xq | - | - | - | - | cuml-gpu 10.2 ms (IDENTICAL/arm -); torch-gpu 2.2 ms (IDENTICAL/arm -) |
| classical | svc | istella | Xq | - | - | - | - | cuml-gpu 4.6 ms (IDENTICAL/arm -) |
| trees | gbdt-categorical | taxi | test | - | - | - | - | catboost-gpu - ms (IDENTICAL/arm -); xgboost-gpu 87.3 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-categorical | taxi | large | - | - | - | - | catboost-gpu - ms (IDENTICAL/arm -); xgboost-gpu 199.2 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | taxi | test | - | - | - | - | catboost-gpu 354.9 ms (IDENTICAL/arm -); xgboost-gpu 10.1 ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | taxi | large | - | - | - | - | catboost-gpu 732.7 ms (IDENTICAL/arm -); xgboost-gpu 19.6 ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | test | - | - | - | - | catboost-gpu 356.8 ms (IDENTICAL/arm -); xgboost-gpu 10.2 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | large | - | - | - | - | catboost-gpu 701.1 ms (IDENTICAL/arm -); xgboost-gpu 18.2 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | taxi | test | - | - | - | - | catboost-gpu 75.6 ms (IDENTICAL/arm -); xgboost-gpu 31.9 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | taxi | large | - | - | - | - | catboost-gpu 116.8 ms (IDENTICAL/arm -); xgboost-gpu 56.0 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-ordered | taxi | test | - | - | - | - | catboost-gpu 80.3 ms (IDENTICAL/arm -) |
| trees | gbdt-ordered | taxi | large | - | - | - | - | catboost-gpu 138.4 ms (IDENTICAL/arm -) |
| trees | gbdt-rank-yetirank | istella | test | - | - | - | - | catboost-gpu 91.4 ms (IDENTICAL/arm -); xgboost-gpu 62.2 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-rank-yetirank | istella | large | - | - | - | - | catboost-gpu 125.5 ms (IDENTICAL/arm -); xgboost-gpu 86.9 ms (IDENTICAL/arm -); lightgbm-cuda - ms (IDENTICAL/arm -) |

## Trees

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: failed, driver rc 0, log `raw/trees/gbdt-categorical.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 26444.4 | 26444.4..26444.4 | 1 | - | - | 4571.4 | 498.0 | auc=0.630387, logloss=0.528535 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 13662.3 | 13662.3..13662.3 | 1 | - | - | 1886.0 | 498.0 | auc=0.631978, logloss=0.528548 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-categorical arms=catboost-gpu,xgboost-gpu leaves=catboost-gpu:96634,xgboost-gpu:102162 spread=0.0541 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `catboost-gpu`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | xgboost-gpu |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "gbtree" |
| bootstrap_type | "No" | - | - |
| class_weight | - | null | - |
| feature_border_type | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "lossguide" |
| leaf_estimation_iterations | 1 | - | - |
| leaf_estimation_method | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | - |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 |
| min_samples_leaf | 1 | 20 | - |
| min_split_gain | - | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 |
| nan_mode | "Min" | - | - |
| random_strength | 0.0 | - | - |
| reg_alpha | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "NewtonL2" | - | - |
| seed | 7 | 7 | 7 |
| subsample | null | 1.0 | 1.0 |

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | - | - | - | 0 | - | - | - | - | COMPARABLE | REFUSED(CatBoostError during warm-up: catboost/libs/model/cuda/evaluator.cpp:22: Model contains categorical features, GPU evaluation impossible) |
| xgboost-gpu | test | 500000 | 87.3 | 87.3..87.3 | 1 | - | - | auc=0.631978, auc_matches_fit=True, logloss=0.528548, logloss_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |
| catboost-gpu | large | - | - | - | 0 | - | - | - | - | COMPARABLE | REFUSED(CatBoostError during warm-up: catboost/libs/model/cuda/evaluator.cpp:22: Model contains categorical features, GPU evaluation impossible) |
| xgboost-gpu | large | 1000000 | 199.2 | 199.2..199.2 | 1 | - | - | - | yes | COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |

inference call, catboost-gpu: catboost predict_proba(int64 categorical frame built in the clock, task_type GPU), column 1

inference call, xgboost-gpu: xgboost XGBClassifier.predict_proba(pandas CategoricalDtype frame built in the clock; inplace_predict takes no category frame here), column 1

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 6393.8 | 6393.8..6393.8 | 1 | - | - | 1576.3 | 496.0 | auc=0.632335, logloss=0.527912 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 3309.5 | 3309.5..3309.5 | 1 | - | - | 1670.3 | 496.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-gpu,xgboost-gpu leaves=catboost-gpu:104203,xgboost-gpu:91234 spread=0.1245 verdict=NOT-COMPARABLE`

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
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | - |
| seed | 7 | 7 |
| subsample | null | 1.0 |

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 500000 | 354.9 | 354.9..354.9 | 1 | - | - | auc=0.632335, auc_matches_fit=True, logloss=0.527912, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 10.1 | 10.1..10.1 | 1 | - | - | auc=0.630969, auc_matches_fit=True, logloss=0.528678, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 732.7 | 732.7..732.7 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 19.6 | 19.6..19.6 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: failed, driver rc 0, log `raw/trees/gbdt-lossguide.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 15380.3 | 15380.3..15380.3 | 1 | - | - | 1670.4 | 496.0 | auc=0.631766, logloss=0.528065 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 6565.0 | 6565.0..6565.0 | 1 | - | - | 1720.3 | 496.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-gpu,xgboost-gpu leaves=catboost-gpu:102046,xgboost-gpu:91234 spread=0.1060 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `catboost-gpu`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | xgboost-gpu |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "gbtree" |
| bootstrap_type | "No" | - | - |
| class_weight | - | null | - |
| feature_border_type | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "lossguide" |
| leaf_estimation_iterations | 1 | - | - |
| leaf_estimation_method | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | - |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 |
| min_samples_leaf | 1 | 20 | - |
| min_split_gain | - | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 |
| nan_mode | "Min" | - | - |
| random_strength | 0.0 | - | - |
| reg_alpha | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "NewtonL2" | - | - |
| seed | 7 | 7 | 7 |
| subsample | null | 1.0 | 1.0 |

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 500000 | 356.8 | 356.8..356.8 | 1 | - | - | auc=0.631766, auc_matches_fit=True, logloss=0.528065, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 10.2 | 10.2..10.2 | 1 | - | - | auc=0.630969, auc_matches_fit=True, logloss=0.528678, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |
| catboost-gpu | large | 1000000 | 701.1 | 701.1..701.1 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 18.2 | 18.2..18.2 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: failed, driver rc 0, log `raw/trees/gbdt-multiclass.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 10221.5 | 10221.5..10221.5 | 1 | - | - | 1717.7 | 610.0 | accuracy=0.599270, mlogloss=1.012704 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 12711.7 | 12711.7..12711.7 | 1 | - | - | 1903.0 | 610.0 | accuracy=0.601200, mlogloss=1.005204 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-gpu,xgboost-gpu leaves=catboost-gpu:128000,xgboost-gpu:99730 spread=0.2209 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `catboost-gpu`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | xgboost-gpu |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "gbtree" |
| bootstrap_type | "No" | - | - |
| class_weight | - | null | - |
| feature_border_type | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "depthwise" |
| leaf_estimation_iterations | 1 | - | - |
| leaf_estimation_method | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "MultiClass" | - | - |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 |
| min_samples_leaf | 1 | 20 | - |
| min_split_gain | - | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 |
| nan_mode | "Min" | - | - |
| random_strength | 1.0 | - | - |
| reg_alpha | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 |
| score_function | "Cosine" | - | - |
| seed | 7 | 7 | 7 |
| subsample | null | 1.0 | 1.0 |

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 500000 | 75.6 | 75.6..75.6 | 1 | - | - | accuracy=0.599270, accuracy_matches_fit=True, mlogloss=1.012704, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 31.9 | 31.9..31.9 | 1 | - | - | accuracy=0.601200, accuracy_matches_fit=True, mlogloss=1.005204, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |
| catboost-gpu | large | 1000000 | 116.8 | 116.8..116.8 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 56.0 | 56.0..56.0 | 1 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:28: Model is not one-dimensional, GPU eva), the (rows, n_classes) probability matrix

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the (rows, n_classes) probability matrix

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-ordered.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 20288.2 | 20288.2..20288.2 | 1 | - | - | 1205.7 | 430.0 | auc=0.628945, logloss=0.528997 | yes | UNKNOWN | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-ordered arms=catboost-gpu leaves=catboost-gpu:92962 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `catboost-gpu`, seed 7): MATCHED

| parameter | catboost-gpu |
|---|---|
| library (source) | catboost (get_params) |
| boosting_type | "Ordered" |
| bootstrap_type | "No" |
| feature_border_type | "GreedyLogSum" |
| fold_len_multiplier | 2.0 |
| fold_permutation_block | 64 |
| grow_policy | "SymmetricTree" |
| leaf_estimation_iterations | 1 |
| leaf_estimation_method | "Newton" |
| learning_rate | 0.1 |
| loss | "Logloss" |
| max_bin | 255 |
| max_depth | 8 |
| max_leaves | 256 |
| min_samples_leaf | 1 |
| n_estimators | 500 |
| nan_mode | "Min" |
| permutation_count | null |
| random_strength | 1.0 |
| reg_lambda | 1.0 |
| scale_pos_weight | 1.3101271632087197 |
| score_function | "Cosine" |
| seed | 7 |
| subsample | null |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 500000 | 80.3 | 80.3..80.3 | 1 | - | - | auc=0.628945, auc_matches_fit=True, logloss=0.528997, logloss_matches_fit=True | yes | UNKNOWN | ok |
| catboost-gpu | large | 1000000 | 138.4 | 138.4..138.4 | 1 | - | - | - | yes | UNKNOWN | ok |

inference call, catboost-gpu: catboost predict_proba(X, task_type GPU, thread_count -1), column 1

### gbdt-rank-yetirank / istella (rows full, shape istellarank-2043304x220)

race: failed, driver rc 0, log `raw/trees/gbdt-rank-yetirank.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | catboost | gpu | opponent | 2955.2 | 2955.2..2955.2 | 1 | - | - | 6207.1 | 556.0 | map=0.814091, ndcg10=0.681202, ndcg5=0.614769 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 5077.0 | 5077.0..5077.0 | 1 | - | - | 6783.5 | 556.0 | map=0.842476, ndcg10=0.725584, ndcg5=0.663588 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-rank-yetirank arms=catboost-gpu,xgboost-gpu leaves=catboost-gpu:6400,xgboost-gpu:6399 spread=0.0002 verdict=COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `catboost-gpu`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | xgboost-gpu |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "gbtree" |
| bootstrap_type | "No" | - | - |
| class_weight | - | null | - |
| feature_border_type | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "depthwise" |
| leaf_estimation_iterations | 1 | - | - |
| leaf_estimation_method | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "YetiRank" | - | - |
| max_bin | 255 | 255 | 255 |
| max_depth | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 |
| min_child_weight | - | 0.001 | 0.0 |
| min_samples_leaf | 1 | 20 | - |
| min_split_gain | - | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 |
| nan_mode | "Min" | - | - |
| random_strength | 1.0 | - | - |
| reg_alpha | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 |
| score_function | "Cosine" | - | - |
| seed | 7 | 7 | 7 |
| subsample | null | 1.0 | 1.0 |

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost-gpu | test | 681250 | 91.4 | 91.4..91.4 | 1 | - | - | map=0.814091, map_matches_fit=True, ndcg10=0.681202, ndcg10_matches_fit=True, ndcg5=0.614769, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-gpu | test | 681250 | 62.2 | 62.2..62.2 | 1 | - | - | map=0.842476, map_matches_fit=True, ndcg10=0.725584, ndcg10_matches_fit=True, ndcg5=0.663588, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |
| catboost-gpu | large | 1000000 | 125.5 | 125.5..125.5 | 1 | - | - | - | yes | COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 86.9 | 86.9..86.9 | 1 | - | - | - | yes | COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |

inference call, catboost-gpu: catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the ranker: a host apply on every box), raw ranking scores

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, raw ranking scores

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.dbscan.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 73482.1 | 73482.1..73482.1 | 1 | - | - | 2599.3 | 1272.0 | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 899.7 | 899.7..899.7 | 1 | - | - | 1866.3 | 526.0 | n_clusters=53, noise_fraction=0.256280, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### kde / istella (rows full, shape 100000x220)

race: done, driver rc 0, log `logs/classical.kde.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 6.6 | 6.6..6.6 | 1 | - | - | 1001.0 | 514.0 | mean_log_likelihood=-222.270582, rows_without_density=0 | yes | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

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

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 509.0 | 509.0..509.0 | 1 | - | - | 4957.5 | 2154.0 | inertia=6.111e+17, n_iter=21 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 849.0 | 849.0..849.0 | 1 | - | - | 3162.6 | 3468.9 | inertia=5.991e+17, n_iter=55 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | 500000 | 5.4 | 5.4..5.4 | 1 | - | - | eval_inertia=1.442e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 1.2 | 1.2..1.2 | 1 | - | - | eval_inertia=1.402e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml KMeans.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### knn / istella (rows full, shape 400000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 78.4 | 78.4..78.4 | 1 | - | - | 1622.3 | 772.0 | recall_at_k=0.976402, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 116.0 | 116.0..116.0 | 1 | - | - | 1223.8 | 3816.8 | recall_at_k=0.981012, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### ols / istella (rows full, shape 2043304x220)

race: failed, driver rc 0, log `logs/classical.ols.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 78.5 | 78.5..78.5 | 1 | - | - | 5075.9 | 2154.0 | finite=True, r2=-11031.855105, rmse=87.647429 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 580.8 | 580.8..580.8 | 1 | - | - | 3013.5 | 8894.4 | finite=False, r2=nan, rmse=nan | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 19.8 | 19.8..19.8 | 1 | - | - | 3044.8 | 3455.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | 500000 | 2.3 | 2.3..2.3 | 1 | - | - | predict_max_rel_err_own_fp64=0.001281, r2_eval=-11032.159107, rmse_eval=87.648636 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.9 | 0.9..0.9 | 1 | - | - | predict_max_rel_err_own_fp64=nan, r2_eval=nan, rmse_eval=nan | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu-eigh | Xq | 500000 | 0.9 | 0.9..0.9 | 1 | - | - | predict_max_rel_err_own_fp64=7.288e-07, r2_eval=0.151604, rmse_eval=0.768590 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml LinearRegression.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 74.6 | 74.6..74.6 | 1 | - | - | 5058.4 | 2190.0 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 17.0 | 17.0..17.0 | 1 | - | - | 2998.2 | 3439.6 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | 500000 | 10.2 | 10.2..10.2 | 1 | - | - | transform_max_rel_err_own_fp64=3.694e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 2.2 | 2.2..2.2 | 1 | - | - | transform_max_rel_err_own_fp64=4.129e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml PCA.transform(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

### svc / istella (rows full, shape 10000x220)

race: done, driver rc 0, log `logs/classical.svc.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 23.0 | 23.0..23.0 | 1 | - | - | 1008.8 | 464.0 | accuracy=0.922200, n_support=2401 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| gamma | 0.004545454545454545 |
| kernel | "rbf" |
| max_iter | -1 |
| seed | 7 |
| tol | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | Xq | 10000 | 4.6 | 4.6..4.6 | 1 | - | - | accuracy_eval=0.922200 | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: cuml SVC.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

## Classical, wave 2

### agglomerative / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.agglomerative.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 54.3 | 54.3..54.3 | 1 | - | - | 1705.1 | 440.0 | n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### arima / synthetic (rows full, shape Yfit 64x2000; Yhold 64x100)

race: done, driver rc 0, log `logs/classical2.arima.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 787.6 | 787.6..787.6 | 1 | - | - | 822.4 | 812.0 | forecast_rmse=1.515427, insample_rmse=0.999338, mean_aic=5680.957154, mean_llf=-2836.478577 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: order=(1,0,1), seasonal_order=(0,0,0,0), trend='c' (cuML fit_intercept=True), maxiter=1000, maximum likelihood. Rows: 64 synthetic ARMA(1,1) series, 2000 fit points, 100 held out. Timed: fit of every series.

mismatch: ours and cuML fit the whole batch in one call; statsmodels fits one series per call (the state-space model, L-BFGS), spread over every core with joblib

mismatch: statsmodels enforce_stationarity and enforce_invertibility at its default (True); ours and cuML have no such parameter

mismatch: seed: no arm has a seed argument (maximum likelihood)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | ? (unreadable (NotImplementedError('ARIMA is unable to be cloned via `get_params` and `set_params`.'))) |
| seed | "none (deterministic)" |

### elasticnet / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 5.8 | 5.8..5.8 | 1 | - | - | 968.0 | 498.0 | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/classical2.ivf.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 301.6 | 301.6..301.6 | 1 | - | - | 1730.7 | 770.0 | recall_at_k=0.999975, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### kernel-ridge / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 324.3 | 324.3..324.3 | 1 | - | - | 1840.5 | 494.0 | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| gamma | 0.004545454545454545 |
| kernel | "rbf" |
| seed | "none (deterministic)" |

### knn-clf / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 40.2 | 40.2..40.2 | 1 | - | - | 2095.5 | 602.0 | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### knn-reg / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 36.6 | 36.6..36.6 | 1 | - | - | 1969.1 | 602.0 | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### lasso / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 935.1 | 935.1..935.1 | 1 | - | - | 2918.9 | 1374.0 | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### linearsvc / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 248.1 | 248.1..248.1 | 1 | - | - | 3028.1 | 1366.0 | accuracy=0.922860 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### linearsvr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 83.7 | 83.7..83.7 | 1 | - | - | 2937.3 | 1374.0 | finite=True, r2=-0.106761, rmse=0.878795 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### logreg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 1854.6 | 1854.6..1854.6 | 1 | - | - | 3047.8 | 1374.0 | accuracy=0.924430, logloss=0.181268, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.ridge.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 48.2 | 48.2..48.2 | 1 | - | - | 2969.7 | 1410.0 | finite=True, r2=-0.251259, rmse=0.934403 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### spectral-embedding / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 137.8 | 137.8..137.8 | 1 | - | - | 1016.9 | 496.0 | trustworthiness_k15=0.882904 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### spectral / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.spectral.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 112.5 | 112.5..112.5 | 1 | - | - | 1939.7 | 490.0 | n_clusters=8, silhouette=0.147757 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### svr / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.svr.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 40.5 | 40.5..40.5 | 1 | - | - | 1813.6 | 462.0 | finite=True, r2=0.318232, rmse=0.680829 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| gamma | 0.004545454545454545 |
| kernel | "rbf" |
| max_iter | -1 |
| seed | "none (deterministic)" |
| tol | 0.001 |

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc 0, log `logs/classical2.tsvd.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 35.3 | 35.3..35.3 | 1 | - | - | 2749.0 | 1314.0 | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001472 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### umap / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.umap.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 502.3 | 502.3..502.3 | 1 | - | - | 1035.8 | 606.0 | trustworthiness_k15=0.979912 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### gemm-bf16 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-bf16.gaussian.shape-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-bf16 | torch | gpu | opponent | 44.0 | 44.0..44.0 | 1 | - | - | 1226.8 | 176.2 | max_rel_err_vs_fp64=0.002764 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 65.8 | 65.8..65.8 | 1 | - | - | 1445.1 | 176.2 | max_rel_err_vs_fp64=0.002764 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-bf16`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-eager-bf16 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm.gaussian.shape-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 50.2 | 50.2..50.2 | 1 | - | - | 933.6 | 200.1 | max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 46.8 | 46.8..46.8 | 1 | - | - | 938.7 | 200.1 | max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 49.2 | 49.2..49.2 | 1 | - | - | 1146.1 | 200.1 | max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 47.6 | 47.6..47.6 | 1 | - | - | 1144.8 | 200.1 | max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 45.3 | 45.3..45.3 | 1 | - | - | 1174.9 | 240.2 | max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 47.4 | 47.4..47.4 | 1 | - | - | 1422.4 | 240.2 | max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

### lm-train-step / bytes (neural shape full: -)

race: failed, driver rc 1, log `logs/neural.lm-train-step.bytes.shape-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |

config: the board's own settings (no NVIDIA harness entry)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

### mamba2-forward / gaussian (neural shape full: -)

race: failed, driver rc 1, log `logs/neural.mamba2-forward.gaussian.shape-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |

config: the board's own settings (no NVIDIA harness entry)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-train-step.gaussian.shape-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.5 | 1.5..1.5 | 1 | - | - | 1015.5 | 16.3 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 1016.6 | 16.3 | loss_first_step=1.160392, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 1104.5 | 16.3 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 1049.0 | 16.3 | loss_first_step=1.160392, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 1229.1 | 16.3 | loss_first_step=1.160498, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.4 | 2.4..2.4 | 1 | - | - | 1295.1 | 16.3 | loss_first_step=1.160498, loss_last_step=1.123462, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### samba-train-step / bytes (neural shape full: -)

race: failed, driver rc 1, log `logs/neural.samba-train-step.bytes.shape-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | UNKNOWN | - | UNKNOWN(no race json, rc 1) (measured this run) |

config: the board's own settings (no NVIDIA harness entry)

parameters: NOT CHECKED (the driver printed no BOARD-PARAMS line)

## Algorithm expansion

### adafactor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adafactor.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 15.3 | 15.3..15.3 | 1 | - | - | 1035.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 169.7 | 169.7..169.7 | 1 | - | - | 1118.9 | 1024.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'beta2_decay': -0.8, 'd': 1.0, 'eps': [None, 0.001], 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| eps | [null, 0.001] | [null, 0.001] |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### adam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adam.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 16.8 | 16.8..16.8 | 1 | - | - | 970.9 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 141.0 | 141.0..141.0 | 1 | - | - | 1441.2 | 896.0 | rel_fro_vs_torch_eager_fp32=4.891e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### adamw / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.adamw.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 18.0 | 18.0..18.0 | 1 | - | - | 970.7 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 147.8 | 147.8..147.8 | 1 | - | - | 1440.7 | 896.0 | rel_fro_vs_torch_eager_fp32=2.203e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'eps': 1e-08, 'lr': 0.001, 'maximize': False, 'weight_decay': 0.01}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| amsgrad | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.01 | 0.01 |

### als / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.als.text.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implicit-gpu | implicit | gpu | opponent | 2560.8 | 2560.8..2560.8 | 1 | - | - | 2485.7 | 506.0 | recall_at_10=0.546926 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### autoarima / taxi-hourly (rows full, shape Yfit 64x1392; Yhold 64x48)

race: done, driver rc 0, log `logs/algos.autoarima.taxi-hourly.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 157302.2 | 157302.2..157302.2 | 1 | - | - | 868.2 | 816.0 | forecast_rmse=106.817545 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### avgpool2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.avgpool2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 711.3 | 541.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | 943.1 | 540.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 711.5 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 892.9 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 711.3 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 887.7 | 541.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None, 'kernel_size': 2, 'padding': 0, 'stride': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### batchnorm2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.batchnorm2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 761.8 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | 965.4 | 246.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | 761.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.6 | 1.6..1.6 | 1 | - | - | 903.3 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | 761.7 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 894.8 | 250.0 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=4.327e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'affine': True, 'eps': 1e-05, 'momentum': 0.1, 'num_features': 64, 'track_running_stats': True}. Rows: None. Timed: None.

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
| torch-eager-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### bernoulli-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bernoulli-nb.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 12.9 | 12.9..12.9 | 1 | - | - | 1172.6 | 494.0 | accuracy=0.755560, logloss=0.557803 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### binarizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.binarizer.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 2.4 | 2.4..2.4 | 1 | - | - | 892.9 | 486.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 11.4 | 11.4..11.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### cagra / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.cagra.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 460.0 | 460.0..460.0 | 1 | - | - | 1410.6 | 526.0 | recall_at_10=0.619300 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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
| cuvs-gpu | Xq | - | 1275.0 | 1275.0..1275.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### categorical-nb / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.categorical-nb.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 13.5 | 13.5..13.5 | 1 | - | - | 1109.7 | 460.0 | accuracy=0.765850, logloss=0.538866 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 6.5 | 6.5..6.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### clip-grad-norm / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.clip-grad-norm.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 991.9 | 128.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 0.5 | 0.5..0.5 | 1 | - | - | 1182.7 | 128.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'error_if_nonfinite': True, 'max_norm': 1.0, 'norm_type': 2.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

### complement-nb / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.complement-nb.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 56.5 | 56.5..56.5 | 1 | - | - | 3934.4 | 1372.0 | accuracy=0.849350, logloss=3.763060 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### complement-nb / text (rows full, shape X 86626x4096; Xq 9626x4096; y 86626; yq 9626)

race: done, driver rc 0, log `logs/algos.complement-nb.text.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 61.1 | 61.1..61.1 | 1 | - | - | 4710.5 | 1946.0 | accuracy=0.983067, logloss=0.559490 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### connected-components / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.connected-components.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cugraph-gpu | cugraph | gpu | opponent | 23.7 | 23.7..23.7 | 1 | - | - | 934.4 | 432.0 | n_components=588 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cugraph-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cugraph-gpu`, seed 7): MATCHED

| parameter | cugraph-gpu |
|---|---|
| library (source) | cugraph (declared) |
| seed | "none (deterministic)" |

### conv2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.conv2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 910.5 | 250.5 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.5 | 2.5..2.5 | 1 | - | - | 1131.1 | 346.8 | max_rel_diff_vs_torch_eager_fp32=0.506768, rel_fro_vs_torch_eager_fp32=5.007e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | - | - | 927.7 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.2 | 2.2..2.2 | 1 | - | - | 1081.7 | 354.2 | max_rel_diff_vs_torch_eager_fp32=382.931903, rel_fro_vs_torch_eager_fp32=0.0003021 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 971.3 | 273.7 | max_rel_diff_vs_torch_eager_fp32=3418.337554, rel_fro_vs_torch_eager_fp32=0.003382 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 1180.3 | 272.7 | max_rel_diff_vs_torch_eager_fp32=3433.596343, rel_fro_vs_torch_eager_fp32=0.003380 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'bias': True, 'dilation': 1, 'groups': 1, 'in_channels': 64, 'kernel_size': 3, 'out_channels': 64, 'padding': 1, 'padding_mode': 'zeros', 'stride': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.2 | 1.2..1.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### dart-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart-reg.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 10334.1 | 10334.1..10334.1 | 1 | - | - | 2005.6 | 504.0 | finite=True, r2=0.577383, rmse=0.543042 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| xgboost-gpu | Xq | - | 146.3 | 146.3..146.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, xgboost-gpu: predict(Xq)(Xq)

### dart / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.dart.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 10004.8 | 10004.8..10004.8 | 1 | - | - | 2043.5 | 504.0 | accuracy=0.954280, logloss=0.115097 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| xgboost-gpu | Xq | - | 157.1 | 157.1..157.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, xgboost-gpu: predict(Xq)(Xq)

### decision-tree-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-clf.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 516.3 | 516.3..516.3 | 1 | - | - | 2957.0 | 1368.0 | accuracy=0.935000, logloss=0.758303 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### decision-tree-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.decision-tree-reg.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 523.1 | 523.1..523.1 | 1 | - | - | 2864.0 | 1364.0 | finite=True, r2=0.379438, rmse=0.658041 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### dropout2d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.dropout2d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | 697.8 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | 916.4 | 247.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | 697.7 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 852.0 | 250.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'p': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-compile-tf32 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| p | 0.1 | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.2 | 0.2..0.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

### embedding / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.embedding.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 838.7 | 782.6 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.1 | 2.1..2.1 | 1 | - | - | 1167.7 | 640.2 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'embedding_dim': 1024, 'max_norm': None, 'norm_type': 2.0, 'num_embeddings': 32768, 'padding_idx': None, 'scale_grad_by_freq': False, 'sparse': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (declared) | torch (declared) |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

### gaussian-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gaussian-nb.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 22.3 | 22.3..22.3 | 1 | - | - | 974.2 | 486.0 | accuracy=0.719810, logloss=1.132317 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 8.5 | 8.5..8.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### gaussian-rp / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.gaussian-rp.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 894.4 | 438.0 | mean_abs_distortion=0.302259 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### gcn / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.gcn.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 12.6 | 12.6..12.6 | 1 | - | - | 1344.8 | 1589.7 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 3.3 | 3.3..3.3 | 1 | - | - | 1281.3 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002327, rel_fro_vs_torch_eager_fp32=9.803e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 10.9 | 10.9..10.9 | 1 | - | - | 1347.7 | 1589.7 | max_rel_diff_vs_torch_eager_fp32=0.001863, rel_fro_vs_torch_eager_fp32=7.223e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 3.7 | 3.7..3.7 | 1 | - | - | 1242.7 | 310.5 | max_rel_diff_vs_torch_eager_fp32=0.002794, rel_fro_vs_torch_eager_fp32=9.801e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 15.9 | 15.9..15.9 | 1 | - | - | 1464.0 | 1874.9 | max_rel_diff_vs_torch_eager_fp32=1509.509282, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 9.3 | 9.3..9.3 | 1 | - | - | 1468.7 | 591.0 | max_rel_diff_vs_torch_eager_fp32=1509.508234, rel_fro_vs_torch_eager_fp32=0.002330 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.5 | 1.5..1.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 1.5 | 1.5..1.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### global-maxpool / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.global-maxpool.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 669.4 | 12.9 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.2 | 1.2..1.2 | 1 | - | - | 904.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 669.8 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 1.1 | 1.1..1.1 | 1 | - | - | 855.8 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 669.4 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.0 | 1.0..1.0 | 1 | - | - | 852.1 | 12.9 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### graphsage / taxi (rows full, shape X 100000x11; indices 1258298; indptr 100001; y 100000)

race: done, driver rc 0, log `logs/algos.graphsage.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.6 | 1.6..1.6 | 1 | - | - | 1181.3 | 289.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | - | - | 1246.5 | 240.0 | max_rel_diff_vs_torch_eager_fp32=0.238419, rel_fro_vs_torch_eager_fp32=6.954e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | - | - | 1182.5 | 289.1 | max_rel_diff_vs_torch_eager_fp32=0.029851, rel_fro_vs_torch_eager_fp32=4.218e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | - | - | 1240.6 | 240.0 | max_rel_diff_vs_torch_eager_fp32=1475.334167, rel_fro_vs_torch_eager_fp32=0.0002145 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 1316.0 | 262.6 | max_rel_diff_vs_torch_eager_fp32=5460.333333, rel_fro_vs_torch_eager_fp32=0.003557 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 1364.9 | 167.2 | max_rel_diff_vs_torch_eager_fp32=4608.154297, rel_fro_vs_torch_eager_fp32=0.003285 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 0.8 | 0.8..0.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### gru-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-clf.taxi-hourly.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 591.3 | 591.3..591.3 | 1 | - | - | 1122.2 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 662.7 | 662.7..662.7 | 1 | - | - | 1173.5 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 641.7 | 641.7..641.7 | 1 | - | - | 1123.2 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 733.4 | 733.4..733.4 | 1 | - | - | 1174.8 | 643.7 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 838.4 | 838.4..838.4 | 1 | - | - | 1487.9 | 340.9 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 789.0 | 789.0..789.0 | 1 | - | - | 1539.0 | 340.9 | accuracy=0.865668 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 2.5 | 2.5..2.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.5 | 2.5..2.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.3 | 2.3..2.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### gru-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.gru-reg.taxi-hourly.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 650.5 | 650.5..650.5 | 1 | - | - | 1137.2 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 859.6 | 859.6..859.6 | 1 | - | - | 1188.0 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 545.3 | 545.3..545.3 | 1 | - | - | 1138.4 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 721.3 | 721.3..721.3 | 1 | - | - | 1190.0 | 643.4 | finite=True, r2=0.748219, rmse=0.544182 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 696.1 | 696.1..696.1 | 1 | - | - | 1534.1 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 762.0 | 762.0..762.0 | 1 | - | - | 1585.0 | 340.6 | finite=True, r2=0.748325, rmse=0.544067 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.6 | 2.6..2.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 2.5 | 2.5..2.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### incremental-pca / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.incremental-pca.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 178.2 | 178.2..178.2 | 1 | - | - | 1176.1 | 518.0 | explained_variance_fraction=0.999995 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 9.1 | 9.1..9.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### ivf-pq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-pq.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 471.5 | 471.5..471.5 | 1 | - | - | 995.4 | 466.0 | recall_at_10=0.976875 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuvs-gpu | Xq | - | 4.0 | 4.0..4.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### ivf-refine / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-refine.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 381.6 | 381.6..381.6 | 1 | - | - | 1026.7 | 468.0 | recall_at_10=0.999050 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuvs-gpu | Xq | - | 57.5 | 57.5..57.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### ivf-sq / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/algos.ivf-sq.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuvs-gpu | cuvs | gpu | opponent | 138.0 | 138.0..138.0 | 1 | - | - | 960.6 | 464.0 | recall_at_10=0.772500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuvs-gpu | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuvs-gpu: search(queries)(Xq)

### kbins / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.kbins.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 26.6 | 26.6..26.6 | 1 | - | - | 952.0 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### kernel-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.kernel-shap.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 595.0 | 595.0..595.0 | 1 | - | - | 1034.4 | 438.0 | rel_error_vs_exact=7.647e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'l1_reg': False, 'link': 'identity', 'n_background': 100, 'nsamples': 2048}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (declared) |
| seed | 7 |

### label-binarizer / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-binarizer.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 56.3 | 56.3..56.3 | 1 | - | - | 3476.3 | 1560.0 | output_shape=100000x259 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 30.1 | 30.1..30.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### label-encoder / taxi (rows full, shape X 1000000x5; Xq 100000x5; lab 1000000; labq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.label-encoder.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 38.9 | 38.9..38.9 | 1 | - | - | 1025.1 | 472.0 | output_shape=100000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 4.1 | 4.1..4.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 4.5 | 4.5..4.5 | 1 | - | - | 1046.3 | 498.0 | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### louvain / istella (rows full, shape X 20000x220; indices 304830; indices2 62848; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.louvain.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cugraph-gpu | cugraph | gpu | opponent | 218.4 | 218.4..218.4 | 1 | - | - | 947.9 | 440.0 | modularity=0.909430, n_communities=41 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### lstm-clf / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-clf.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 516.0 | 516.0..516.0 | 1 | - | - | 1122.6 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 931.7 | 931.7..931.7 | 1 | - | - | 1173.6 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 532.1 | 532.1..532.1 | 1 | - | - | 1124.1 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 675.9 | 675.9..675.9 | 1 | - | - | 1175.8 | 695.4 | accuracy=0.968696 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 743.9 | 743.9..743.9 | 1 | - | - | 1490.9 | 367.2 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 838.2 | 838.2..838.2 | 1 | - | - | 1542.0 | 367.2 | accuracy=0.968913 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 2.3 | 2.3..2.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.5 | 2.5..2.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstm-reg / synthetic (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.lstm-reg.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 826.4 | 826.4..826.4 | 1 | - | - | 1131.8 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 564.5 | 564.5..564.5 | 1 | - | - | 1183.3 | 695.1 | finite=True, r2=0.981013, rmse=0.159641 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 801.6 | 801.6..801.6 | 1 | - | - | 1133.8 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 797.0 | 797.0..797.0 | 1 | - | - | 1184.9 | 695.1 | finite=True, r2=0.981013, rmse=0.159642 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 978.7 | 978.7..978.7 | 1 | - | - | 1536.5 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 857.8 | 857.8..857.8 | 1 | - | - | 1587.9 | 366.9 | finite=True, r2=0.980998, rmse=0.159706 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 3.3 | 3.3..3.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 4.2 | 4.2..4.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### lstsq / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: failed, driver rc 0, log `logs/algos.lstsq.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 220.0 | 220.0..220.0 | 1 | - | - | 1783.9 | 3516.2 | relative_residual=nan | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 244.2 | 244.2..244.2 | 1 | - | - | 2740.5 | 9158.0 | relative_residual=0.849957 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### lu-factor / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.lu-factor.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 36.1 | 36.1..36.1 | 1 | - | - | 824.5 | 528.3 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 39.5 | 39.5..39.5 | 1 | - | - | 875.6 | 1108.0 | relative_residual=3.386e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### maxabs-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.maxabs-scaler.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 32.4 | 32.4..32.4 | 1 | - | - | 3171.9 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 6.1 | 6.1..6.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### maxpool1d / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.maxpool1d.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | - | - | 703.9 | 288.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.6 | 1.6..1.6 | 1 | - | - | 946.1 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.3 | 1.3..1.3 | 1 | - | - | 703.6 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.7 | 2.7..2.7 | 1 | - | - | 894.9 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 3.8 | 3.8..3.8 | 1 | - | - | 703.6 | 288.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | - | - | 890.2 | 232.0 | max_rel_diff_vs_torch_eager_fp32=0.000000, rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 0.3 | 0.3..0.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### minmax-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minmax-scaler.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 41.6 | 41.6..41.6 | 1 | - | - | 3013.1 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 2.7 | 2.7..2.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### moe / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.moe.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 12.0 | 12.0..12.0 | 1 | - | - | 1008.6 | 469.1 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 11.9 | 11.9..11.9 | 1 | - | - | 1273.6 | 446.7 | max_rel_diff_vs_torch_eager_fp32=0.026193, rel_fro_vs_torch_eager_fp32=1.548e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 9.8 | 9.8..9.8 | 1 | - | - | 1004.4 | 469.1 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 5.1 | 5.1..5.1 | 1 | - | - | 1220.6 | 478.7 | max_rel_diff_vs_torch_eager_fp32=1004.691291, rel_fro_vs_torch_eager_fp32=0.017534 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.3 | 4.3..4.3 | 1 | - | - | 1129.6 | 574.3 | max_rel_diff_vs_torch_eager_fp32=22834.612745, rel_fro_vs_torch_eager_fp32=0.055222 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.4 | 4.4..4.4 | 1 | - | - | 1397.4 | 433.6 | max_rel_diff_vs_torch_eager_fp32=22851.987745, rel_fro_vs_torch_eager_fp32=0.055199 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-tf32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'hidden_size': 1024, 'intermediate_size': 2816, 'norm_topk_prob': True, 'num_experts': 8, 'top_k': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| hidden_size | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| seed | 7 | 7 | 7 | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | Xq | - | 11.2 | 11.2..11.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 12.5 | 12.5..12.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 4.4 | 4.4..4.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 3.4 | 3.4..3.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.7 | 3.7..3.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: forward(x) (no autograd)(Xq)

inference call, torch-compile-fp32: forward(x) (no autograd)(Xq)

inference call, torch-eager-tf32: forward(x) (no autograd)(Xq)

inference call, torch-compile-tf32: forward(x) (no autograd)(Xq)

inference call, torch-eager-bf16: forward(x) (no autograd)(Xq)

inference call, torch-compile-bf16: forward(x) (no autograd)(Xq)

### multinomial-nb / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.multinomial-nb.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 72.6 | 72.6..72.6 | 1 | - | - | 1077.0 | 496.0 | accuracy=0.723160, logloss=0.590750 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 3.5 | 3.5..3.5 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### nadam / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.nadam.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 20.8 | 20.8..20.8 | 1 | - | - | 971.0 | 960.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 174.5 | 174.5..174.5 | 1 | - | - | 1443.1 | 896.0 | rel_fro_vs_torch_eager_fp32=1.498e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'betas': [0.9, 0.999], 'decoupled_weight_decay': False, 'eps': 1e-08, 'lr': 0.001, 'momentum_decay': 0.004, 'weight_decay': 0.0}. Rows: None. Timed: None.

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

### normalizer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.normalizer.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 0.6 | 0.6..0.6 | 1 | - | - | 935.1 | 496.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### onehot / taxi (rows full, shape X 1000000x5; Xq 100000x5; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.onehot.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 173.9 | 173.9..173.9 | 1 | - | - | 1557.8 | 656.0 | output_shape=100000x508 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 44.8 | 44.8..44.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### pagerank / taxi (rows full, shape X 20000x11; indices 253708; indices2 54528; indptr 20001; indptr2 20001; y 20000)

race: done, driver rc 0, log `logs/algos.pagerank.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cugraph-gpu | cugraph | gpu | opponent | 5.0 | 5.0..5.0 | 1 | - | - | 1023.2 | 438.0 | sum=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### permutation-shap / taxi (rows full, shape X 100000x11; Xq 100x11; y 100000; yq 100)

race: done, driver rc 0, log `logs/algos.permutation-shap.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 319.6 | 319.6..319.6 | 1 | - | - | 956.8 | 438.0 | rel_error_vs_exact=1.701e-07 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'n_background': 100, 'npermutations': 10}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `cuml-gpu`, seed 7): MATCHED

| parameter | cuml-gpu |
|---|---|
| library (source) | cuml (declared) |
| seed | 7 |

### poly-features / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poly-features.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 1.5 | 1.5..1.5 | 1 | - | - | 944.4 | 510.0 | output_shape=100000x77 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 4.6 | 4.6..4.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### power-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.power-transformer.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 3161.1 | 3161.1..3161.1 | 1 | - | - | 942.6 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

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
| cuml-gpu | Xq | - | 13.7 | 13.7..13.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### qn-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qn-reg.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 6.3 | 6.3..6.3 | 1 | - | - | 977.5 | 484.0 | finite=True, r2=0.908983, rmse=4.805040 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### qr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.qr.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 7.0 | 7.0..7.0 | 1 | - | - | 831.1 | 168.1 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 6.8 | 6.8..6.8 | 1 | - | - | 653.7 | 726.0 | relative_gram_difference=5.971e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'mode': 'reduced'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### quantile-transformer / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile-transformer.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 258.6 | 258.6..258.6 | 1 | - | - | 933.8 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 122.2 | 122.2..122.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### randomized-svd / taxi (rows full, shape X 900000x11; Xq 100000x11)

race: done, driver rc 0, log `logs/algos.randomized-svd.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 32.0 | 32.0..32.0 | 1 | - | - | 838.9 | 198.1 | relative_reconstruction_error=0.027197 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### rmsprop / synthetic (rows full, shape -)

race: done, driver rc 0, log `logs/algos.rmsprop.synthetic.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 13.3 | 13.3..13.3 | 1 | - | - | 958.5 | 896.0 | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 118.3 | 118.3..118.3 | 1 | - | - | 1308.8 | 832.0 | rel_fro_vs_torch_eager_fp32=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-eager-fp32, torch-compile-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: {'alpha': 0.99, 'centered': False, 'eps': 1e-08, 'lr': 0.001, 'momentum': 0.0, 'weight_decay': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-eager-fp32`, seed 7): MATCHED

| parameter | torch-compile-fp32 | torch-eager-fp32 |
|---|---||---|---|
| library (source) | torch (optimizer.defaults) | torch (optimizer.defaults) |
| alpha | 0.99 | 0.99 |
| eps | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 |
| momentum | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weight_decay | 0.0 | 0.0 |

### rnn-clf / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-clf.taxi-hourly.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 507.7 | 507.7..507.7 | 1 | - | - | 1122.3 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 667.6 | 667.6..667.6 | 1 | - | - | 1174.0 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 518.2 | 518.2..518.2 | 1 | - | - | 1123.5 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 597.0 | 597.0..597.0 | 1 | - | - | 1174.7 | 404.2 | accuracy=0.868056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 786.6 | 786.6..786.6 | 1 | - | - | 1459.1 | 220.9 | accuracy=0.868001 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 642.8 | 642.8..642.8 | 1 | - | - | 1511.1 | 220.9 | accuracy=0.868001 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 1.2 | 1.2..1.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### rnn-reg / taxi-hourly (rows full, shape X 72192x24x1; Xq 18432x24x1; y 72192; yq 18432)

race: done, driver rc 0, log `logs/algos.rnn-reg.taxi-hourly.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-eager-fp32 | torch | gpu | opponent | 664.7 | 664.7..664.7 | 1 | - | - | 1138.1 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 753.0 | 753.0..753.0 | 1 | - | - | 1189.1 | 403.9 | finite=True, r2=0.738796, rmse=0.554271 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 724.2 | 724.2..724.2 | 1 | - | - | 1133.0 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 633.1 | 633.1..633.1 | 1 | - | - | 1184.7 | 403.9 | finite=True, r2=0.738800, rmse=0.554267 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 612.7 | 612.7..612.7 | 1 | - | - | 1505.1 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 877.8 | 877.8..877.8 | 1 | - | - | 1556.6 | 220.5 | finite=True, r2=0.738906, rmse=0.554155 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| torch-eager-fp32 | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-fp32 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-tf32 | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-tf32 | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-eager-bf16 | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| torch-compile-bf16 | Xq | - | 3.1 | 3.1..3.1 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, torch-eager-fp32: predict(Xq)(Xq)

inference call, torch-compile-fp32: predict(Xq)(Xq)

inference call, torch-eager-tf32: predict(Xq)(Xq)

inference call, torch-compile-tf32: predict(Xq)(Xq)

inference call, torch-eager-bf16: predict(Xq)(Xq)

inference call, torch-compile-bf16: predict(Xq)(Xq)

### robust-scaler / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.robust-scaler.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 34.3 | 34.3..34.3 | 1 | - | - | 947.6 | 488.0 | output_shape=100000x11 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### sgd-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 1543.9 | 1543.9..1543.9 | 1 | - | - | 1092.9 | 498.0 | accuracy=0.703120 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 0.9 | 0.9..0.9 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: predict(Xq)(Xq)

### sgd-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.taxi.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 1579.7 | 1579.7..1579.7 | 1 | - | - | 971.4 | 498.0 | finite=True, r2=0.908979, rmse=4.805168 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

### simple-imputer / istella (rows full, shape X 1000000x220; X_true 1000000x220; Xq 100000x220; Xq_true 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.simple-imputer.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 250.6 | 250.6..250.6 | 1 | - | - | 4914.9 | 1442.0 | masked_rmse=346849.129968 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 38.2 | 38.2..38.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### sparse-rp / istella (rows full, shape X 900000x220; Xq 100000x220)

race: done, driver rc 0, log `logs/algos.sparse-rp.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 108.4 | 108.4..108.4 | 1 | - | - | 1993.6 | 430.0 | mean_abs_distortion=0.876709 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 19.6 | 19.6..19.6 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### standard-scaler / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.standard-scaler.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 421.3 | 421.3..421.3 | 1 | - | - | 3012.0 | 1442.0 | output_shape=100000x220 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### svd / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.svd.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch-gpu | torch | gpu | opponent | 266.0 | 266.0..266.0 | 1 | - | - | 1883.3 | 5176.3 | max_rel_singular_value_error=8.068917, relative_reconstruction_error_100k_rows=2.861e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cupy-gpu | cupy | gpu | opponent | 242.8 | 242.8..242.8 | 1 | - | - | 2666.2 | 9146.0 | max_rel_singular_value_error=10270.720886, relative_reconstruction_error_100k_rows=2.385e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

memory, cupy-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: {'full_matrices': False}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `torch-gpu`, seed 7): MATCHED

| parameter | cupy-gpu | torch-gpu |
|---|---||---|---|
| library (source) | cupy (declared) | torch (declared) |
| seed | "none (deterministic)" | "none (deterministic)" |

### svgp / istella (rows full, shape X 100000x220; Xq 20000x220; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.svgp.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpytorch-gpu | gpytorch | gpu | opponent | 36.2 | 36.2..36.2 | 1 | - | - | 2157.6 | 1492.0 | finite=True, r2=-0.106040, rmse=0.878383 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| gpytorch-gpu | Xq | - | 9.2 | 9.2..9.2 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, gpytorch-gpu: predict(Xq)(Xq)

### target-encoder / istella (rows full, shape X 1000000x8; Xq 100000x8; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.target-encoder.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 530.1 | 530.1..530.1 | 1 | - | - | 1169.3 | 684.0 | output_shape=100000x8 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| cuml-gpu | Xq | - | 53.4 | 53.4..53.4 | 1 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, cuml-gpu: transform(Xq)(Xq)

### tree-shap / istella (rows full, shape X 100000x220; Xq 10000x220; y 100000; yq 10000)

race: done, driver rc 0, log `logs/algos.tree-shap.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost-gpu | xgboost | gpu | opponent | 30.2 | 30.2..30.2 | 1 | - | - | 1535.8 | 530.0 | max_additivity_error=1.956e-06 | - | SPAN-ASYMMETRIC(fit_before_its_clock) | - | ok (measured this run) |

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

### tsne / istella (rows full, shape X 20000x220; Xq 2000x220)

race: done, driver rc 0, log `logs/algos.tsne.istella.rows-full.log`, ran on 24a11adce16e

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuml-gpu | cuml | gpu | opponent | 467.7 | 467.7..467.7 | 1 | - | - | 966.6 | 464.0 | trustworthiness_k15=0.990033 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

