# mojolearn benchmark board

Generated 2026-09-30T03:00:25Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | nvidia / cuda |
| GPU | NVIDIA L40S |
| GPU driver | 580.159.03 |
| CPU | AMD EPYC 9554 64-Core Processor (256 logical cores) |
| memory bytes | 1081799102464 |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.10 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-manylinux_2_35_x86_64.whl, sha256 bafc6196927055d47f9113bf90ed378fdcf895ce215aceafa24e8787cd974641) |
| script commit | 4f3e685348ddc41ec22ce00c53187c3528d929b7 |
| patch sync | synced commit 4f3e685348ddc41ec22ce00c53187c3528d929b7 over base f4e78b35406eda59f8417b3acaa27c2830dca16c, patch sha256 47576fc48f13c4cd6cc1e4f130be92eb24c82c4aaee319912fb27d47caafc719 |
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
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. The `*-infer` lanes are the CPU *Inference classes and race `torch-cpu-*` arms. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours. On NVIDIA the Mamba forward lanes add mamba_ssm's own fused kernels (the deployment path, our weights loaded): `mamba-ssm-fp32` (TF32 off, Triton fp32 dots IEEE) and `mamba-ssm-tf32` (TF32 on, Triton's default tf32 dots; ANOTHER PRECISION).
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU tier (`mojolearn CPU IDENTICAL`, arm `ours-cpu`): the same public estimator in a worker started under MOJOLEARN_VENDOR=cpu, the wheel's CPU switch (no GPU set loads; the host bindings answer, IDENTICAL only), read back as vendor cpu or refused by name. It races in the same rounds as every arm; `ours CPU / arm` is its median over each opponent's. `bits_equal_vs_ours_identical` compares its output with our GPU IDENTICAL arm's, bit for bit. Our CPU and GPU times are never divided by each other here.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 464 planned, 139 done, 0 failed, 325 pending. Cells: 416 (REFUSED 45, ok 371).

Inference cells: 220 (REFUSED 4, UNKNOWN 12, ok 204).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | ard | istella | r2 (higher is better) | - | -0.122912 | - | sklearn-cpu 0.327402 |
| algos | ard | istella | rmse (lower is better) | - | 0.885183 | - | sklearn-cpu 0.685075 |
| algos | ard | taxi | r2 (higher is better) | - | 0.909193 | - | sklearn-cpu 0.909190 |
| algos | ard | taxi | rmse (lower is better) | - | 4.799513 | - | sklearn-cpu 4.799574 |
| algos | bayesian-ridge | istella | r2 (higher is better) | - | -41643.670747 | - | sklearn-cpu -395441.818968 |
| algos | bayesian-ridge | istella | rmse (lower is better) | - | 170.466932 | - | sklearn-cpu 525.293799 |
| algos | bayesian-ridge | taxi | r2 (higher is better) | - | 0.908981 | - | sklearn-cpu 0.908983 |
| algos | bayesian-ridge | taxi | rmse (lower is better) | - | 4.805109 | - | sklearn-cpu 4.805056 |
| algos | enet-cv | istella | r2 (higher is better) | - | 0.316583 | - | sklearn-cpu 0.326805 |
| algos | enet-cv | istella | rmse (lower is better) | - | 0.690563 | - | sklearn-cpu 0.685379 |
| algos | enet-cv | taxi | r2 (higher is better) | - | 0.909002 | - | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | - | 4.804540 | - | sklearn-cpu 4.804486 |
| algos | gamma | istella | r2 (higher is better) | - | 0.027316 | - | sklearn-cpu 0.181278 |
| algos | gamma | istella | rmse (lower is better) | - | 0.823846 | - | sklearn-cpu 0.755838 |
| algos | gamma | taxi | r2 (higher is better) | - | -232.449719 | - | sklearn-cpu -232.959247 |
| algos | gamma | taxi | rmse (lower is better) | - | 243.351187 | - | sklearn-cpu 243.616612 |
| algos | huber | istella | r2 (higher is better) | - | -0.009166 | - | sklearn-cpu -0.010176 |
| algos | huber | istella | rmse (lower is better) | - | 0.839154 | - | sklearn-cpu 0.839574 |
| algos | huber | taxi | r2 (higher is better) | - | 0.900215 | - | sklearn-cpu 0.900215 |
| algos | huber | taxi | rmse (lower is better) | - | 5.031176 | - | sklearn-cpu 5.031163 |
| algos | lars | istella | r2 (higher is better) | - | -1.360350 | - | sklearn-cpu -4.245e+13; cuml-gpu 0.328088 |
| algos | lars | istella | rmse (lower is better) | - | 1.283360 | - | sklearn-cpu 5.442e+06; cuml-gpu 0.684726 |
| algos | lars | taxi | r2 (higher is better) | - | 0.908981 | - | sklearn-cpu 0.908983; cuml-gpu 0.908983 |
| algos | lars | taxi | rmse (lower is better) | - | 4.805109 | - | sklearn-cpu 4.805055; cuml-gpu 4.805052 |
| algos | lasso-cv | istella | r2 (higher is better) | - | 0.310329 | - | sklearn-cpu 0.325507 |
| algos | lasso-cv | istella | rmse (lower is better) | - | 0.693715 | - | sklearn-cpu 0.686040 |
| algos | lasso-cv | taxi | r2 (higher is better) | - | 0.909059 | - | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | - | 4.803051 | - | sklearn-cpu 4.803593 |
| algos | lasso-lars | istella | r2 (higher is better) | - | 0.310330 | - | sklearn-cpu 0.310816 |
| algos | lasso-lars | istella | rmse (lower is better) | - | 0.693715 | - | sklearn-cpu 0.693470 |
| algos | lasso-lars | taxi | r2 (higher is better) | - | 0.908996 | - | sklearn-cpu 0.908998 |
| algos | lasso-lars | taxi | rmse (lower is better) | - | 4.804699 | - | sklearn-cpu 4.804667 |
| algos | pa-clf | istella | accuracy (higher is better) | - | 0.903700 | - | sklearn-cpu 0.890480 |
| algos | pa-clf | taxi | accuracy (higher is better) | - | 0.583830 | - | sklearn-cpu 0.744740 |
| algos | pa-reg | istella | r2 (higher is better) | - | -0.282312 | - | sklearn-cpu -0.128155 |
| algos | pa-reg | istella | rmse (lower is better) | - | 0.945926 | - | sklearn-cpu 0.887248 |
| algos | pa-reg | taxi | r2 (higher is better) | - | 0.853920 | - | sklearn-cpu 0.795107 |
| algos | pa-reg | taxi | rmse (lower is better) | - | 6.087406 | - | sklearn-cpu 7.209421 |
| algos | perceptron | istella | accuracy (higher is better) | - | 0.882480 | - | sklearn-cpu 0.896130 |
| algos | perceptron | taxi | accuracy (higher is better) | - | 0.465380 | - | sklearn-cpu 0.750520 |
| algos | poisson | istella | r2 (higher is better) | - | 0.340827 | - | sklearn-cpu 0.255075 |
| algos | poisson | istella | rmse (lower is better) | - | 0.678204 | - | sklearn-cpu 0.720969 |
| algos | poisson | taxi | r2 (higher is better) | - | 0.035965 | - | sklearn-cpu 0.036205 |
| algos | poisson | taxi | rmse (lower is better) | - | 15.638069 | - | sklearn-cpu 15.636127 |
| algos | quantile | istella | r2 (higher is better) | - | nan | - | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | - | nan | - | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | - | 0.900039 | - | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | - | 5.035615 | - | sklearn-cpu 5.044706 |
| algos | ridge-clf | istella | accuracy (higher is better) | - | 0.894330 | - | sklearn-cpu 0.910540 |
| algos | ridge-clf | taxi | accuracy (higher is better) | - | 0.763570 | - | sklearn-cpu 0.763570 |
| algos | ridge-cv | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.908983 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.805055 |
| algos | sgd-clf | istella | accuracy (higher is better) | - | 0.901150 | - | sklearn-cpu 0.910200; cuml-gpu 0.809350 |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | 0.766410 | - | sklearn-cpu 0.752520; cuml-gpu 0.703120 |
| algos | sgd-ocsvm | istella | fraction_flagged | - | 0.000000 | - | sklearn-cpu 0.093340 |
| algos | sgd-ocsvm | istella | jaccard_vs_sklearn | - | 0.000000 | - | sklearn-cpu 1.000000 |
| algos | sgd-ocsvm | taxi | fraction_flagged | - | 0.000000 | - | sklearn-cpu 0.007020 |
| algos | sgd-ocsvm | taxi | jaccard_vs_sklearn | - | 0.000000 | - | sklearn-cpu 1.000000 |
| algos | sgd-reg | istella | r2 (higher is better) | - | -3.459e+24 | - | sklearn-cpu -2.197e+24; cuml-gpu 0.327768 |
| algos | sgd-reg | istella | rmse (lower is better) | - | 1.554e+12 | - | sklearn-cpu 1.238e+12; cuml-gpu 0.684889 |
| algos | sgd-reg | taxi | r2 (higher is better) | - | 0.868127 | - | sklearn-cpu 0.880681; cuml-gpu 0.908979 |
| algos | sgd-reg | taxi | rmse (lower is better) | - | 5.783813 | - | sklearn-cpu 5.501638; cuml-gpu 4.805168 |
| algos | tweedie | istella | r2 (higher is better) | - | -0.089414 | - | sklearn-cpu -21.706369 |
| algos | tweedie | istella | rmse (lower is better) | - | 0.871880 | - | sklearn-cpu 3.980469 |
| algos | tweedie | taxi | r2 (higher is better) | - | -10.070851 | - | sklearn-cpu -10.073629 |
| algos | tweedie | taxi | rmse (lower is better) | - | 52.994073 | - | sklearn-cpu 53.000721 |
| classical | dbscan | istella | n_clusters | - | 40131 | - | cuml-gpu 40131 |
| classical | dbscan | istella | noise_fraction | - | 0.219391 | - | cuml-gpu 0.219391 |
| classical | dbscan | istella | rows | - | 1000000 | - | cuml-gpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | - | - | - | cuml-gpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | - | - | - | cuml-gpu 1.000000 |
| classical | dbscan | taxi | n_clusters | - | - | - | cuml-gpu 36 |
| classical | dbscan | taxi | noise_fraction | - | - | - | cuml-gpu 0.000174 |
| classical | dbscan | taxi | rows | - | - | - | cuml-gpu 1000000 |
| classical | hdbscan | istella | n_clusters | - | - | - | cuml-gpu 53 |
| classical | hdbscan | istella | noise_fraction | - | - | - | cuml-gpu 0.256280 |
| classical | hdbscan | istella | rows | - | - | - | cuml-gpu 100000 |
| classical | hdbscan | taxi | n_clusters | - | - | - | cuml-gpu 159 |
| classical | hdbscan | taxi | noise_fraction | - | - | - | cuml-gpu 0.130970 |
| classical | hdbscan | taxi | rows | - | - | - | cuml-gpu 100000 |
| classical | kde | istella | mean_log_likelihood (higher is better) | - | -222.270586 | - | cuml-gpu -222.270582 |
| classical | kde | istella | rows_without_density | - | 0 | - | cuml-gpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | - | -14.826460 | - | cuml-gpu -14.826437 |
| classical | kde | taxi | rows_without_density | - | 0 | - | cuml-gpu 0 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | - | cuml-gpu 6.111e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | - | cuml-gpu 1.009914; torch-gpu 0.990156 |
| classical | kmeans | istella | n_iter | - | 33 | - | cuml-gpu 20; torch-gpu 55 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | - | cuml-gpu 3.093e+08; torch-gpu 3.129e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | - | cuml-gpu 1.000059; torch-gpu 1.011537 |
| classical | kmeans | taxi | n_iter | - | 91 | - | cuml-gpu 42; torch-gpu 85 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | - | cuml-gpu 0.976402; torch-gpu 0.981012 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | - | cuml-gpu 0; torch-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | - | cuml-gpu 0.999742; torch-gpu 0.999773 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | - | cuml-gpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.331944 | - | cuml-gpu -11031.855105; torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | 0.682027 | - | cuml-gpu 87.647429; torch-gpu nan; torch-gpu-eigh 0.768590 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908837 | - | cuml-gpu 0.908836; torch-gpu 0.908836; torch-gpu-eigh 0.908836 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696466 | - | cuml-gpu 4.696488; torch-gpu 4.696480; torch-gpu-eigh 4.696490 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | - | torch-gpu 1.000000; cuml-gpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999996 | - | torch-gpu 0.999996; cuml-gpu 0.999996 |
| classical | svc | istella | accuracy (higher is better) | - | 0.922200 | - | cuml-gpu 0.922200 |
| classical | svc | istella | n_support | - | 2400 | - | cuml-gpu 2401 |
| classical | svc | taxi | accuracy (higher is better) | - | 0.767500 | - | cuml-gpu 0.767500 |
| classical | svc | taxi | n_support | - | 5527 | - | cuml-gpu 5541 |
| classical2 | agglomerative | istella | n_clusters | - | 8 | - | cuml-gpu 8 |
| classical2 | agglomerative | istella | silhouette (higher is better) | - | 0.716728 | - | cuml-gpu 0.716728 |
| classical2 | agglomerative | istella | ari_vs_ours (1 is our partition exactly) | - | - | - | cuml-gpu 1.000000 |
| classical2 | agglomerative | taxi | n_clusters | - | 8 | - | cuml-gpu 8 |
| classical2 | agglomerative | taxi | silhouette (higher is better) | - | 0.685524 | - | cuml-gpu 0.685524 |
| classical2 | agglomerative | taxi | ari_vs_ours (1 is our partition exactly) | - | - | - | cuml-gpu 1.000000 |
| classical2 | arima | synthetic | forecast_rmse (lower is better) | - | 1.515518 | - | cuml-gpu 1.515427; statsmodels-cpu 1.515423 |
| classical2 | arima | synthetic | insample_rmse (lower is better) | - | 0.999342 | - | cuml-gpu 0.999338; statsmodels-cpu 0.999338 |
| classical2 | arima | synthetic | mean_aic (lower is better) | - | 5680.976967 | - | cuml-gpu 5680.957154; statsmodels-cpu 5680.957160 |
| classical2 | arima | synthetic | mean_llf (higher is better) | - | -2836.488483 | - | cuml-gpu -2836.478577; statsmodels-cpu -2836.478580 |
| classical2 | elasticnet | istella | r2 (higher is better) | - | 0.260922 | - | cuml-gpu 0.260922 |
| classical2 | elasticnet | istella | rmse (lower is better) | - | 0.718134 | - | cuml-gpu 0.718134 |
| classical2 | elasticnet | taxi | r2 (higher is better) | - | 0.907378 | - | cuml-gpu 0.907378 |
| classical2 | elasticnet | taxi | rmse (lower is better) | - | 4.847224 | - | cuml-gpu 4.847224 |
| classical2 | ets | synthetic | forecast_rmse (lower is better) | - | 0.984392 | - | cuml-gpu 1.073814; statsmodels-cpu 0.984664 |
| classical2 | ets | synthetic | insample_rmse (lower is better) | - | 0.990971 | - | cuml-gpu -; statsmodels-cpu 0.992056 |
| classical2 | gmm | istella | bic (lower is better) | - | -3.851e+07 | - | sklearn-cpu -3.901e+07 |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | - | 200.794403 | - | sklearn-cpu 200.768138 |
| classical2 | gmm | istella | n_iter | - | 24 | - | sklearn-cpu 39 |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.67e+06 | - | sklearn-cpu - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.861940 | - | sklearn-cpu - |
| classical2 | gmm | taxi | n_iter | - | 32 | - | sklearn-cpu - |
| classical2 | gpc | istella | accuracy (higher is better) | - | 0.901333 | - | sklearn-cpu 0.901333 |
| classical2 | gpc | istella | logloss (lower is better) | - | 0.232590 | - | sklearn-cpu 0.232597 |
| classical2 | gpc | istella | nonfinite_proba_rows | - | 0 | - | sklearn-cpu 0 |
| classical2 | gpc | taxi | accuracy (higher is better) | - | 0.761000 | - | sklearn-cpu 0.761000 |
| classical2 | gpc | taxi | logloss (lower is better) | - | 0.541286 | - | sklearn-cpu 0.541358 |
| classical2 | gpc | taxi | nonfinite_proba_rows | - | 0 | - | sklearn-cpu 0 |
| classical2 | gpr | istella | mean_log_predictive_density (higher is better) | - | -9.285754 | - | sklearn-cpu -9.287148 |
| classical2 | gpr | istella | r2 (higher is better) | - | 0.235346 | - | sklearn-cpu 0.235368 |
| classical2 | gpr | istella | rmse (lower is better) | - | 0.760439 | - | sklearn-cpu 0.760428 |
| classical2 | gpr | taxi | mean_log_predictive_density (higher is better) | - | -311.458394 | - | sklearn-cpu -311.539594 |
| classical2 | gpr | taxi | r2 (higher is better) | - | 0.889630 | - | sklearn-cpu 0.889629 |
| classical2 | gpr | taxi | rmse (lower is better) | - | 5.041639 | - | sklearn-cpu 5.041653 |
| classical2 | kernel-ridge | istella | r2 (higher is better) | - | 0.385427 | - | cuml-gpu 0.385427 |
| classical2 | kernel-ridge | istella | rmse (lower is better) | - | 0.646407 | - | cuml-gpu 0.646407 |
| classical2 | kernel-ridge | taxi | r2 (higher is better) | - | 0.726543 | - | cuml-gpu 0.726543 |
| classical2 | kernel-ridge | taxi | rmse (lower is better) | - | 8.330373 | - | cuml-gpu 8.330375 |
| classical2 | knn-clf | istella | accuracy (higher is better) | - | 0.926250 | - | cuml-gpu 0.926250 |
| classical2 | knn-clf | taxi | accuracy (higher is better) | - | 0.741750 | - | cuml-gpu 0.741750 |
| classical2 | knn-reg | istella | r2 (higher is better) | - | 0.418145 | - | cuml-gpu 0.418145 |
| classical2 | knn-reg | istella | rmse (lower is better) | - | 0.625388 | - | cuml-gpu 0.625388 |
| classical2 | knn-reg | taxi | r2 (higher is better) | - | 0.937323 | - | cuml-gpu 0.937323 |
| classical2 | knn-reg | taxi | rmse (lower is better) | - | 3.842028 | - | cuml-gpu 3.842028 |
| classical2 | lasso | istella | r2 (higher is better) | - | 0.310837 | - | cuml-gpu 0.310837 |
| classical2 | lasso | istella | rmse (lower is better) | - | 0.693460 | - | cuml-gpu 0.693460 |
| classical2 | lasso | taxi | r2 (higher is better) | - | 0.908995 | - | cuml-gpu 0.908995 |
| classical2 | lasso | taxi | rmse (lower is better) | - | 4.804745 | - | cuml-gpu 4.804745 |
| classical2 | linearsvc | istella | accuracy (higher is better) | - | 0.923480 | - | cuml-gpu 0.922880 |
| classical2 | linearsvc | taxi | accuracy (higher is better) | - | 0.763330 | - | cuml-gpu 0.762990 |
| classical2 | linearsvr | istella | r2 (higher is better) | - | -0.106754 | - | cuml-gpu -0.106761 |
| classical2 | linearsvr | istella | rmse (lower is better) | - | 0.878792 | - | cuml-gpu 0.878795 |
| classical2 | linearsvr | taxi | r2 (higher is better) | - | 0.899813 | - | cuml-gpu 0.899814 |
| classical2 | linearsvr | taxi | rmse (lower is better) | - | 5.041302 | - | cuml-gpu 5.041286 |
| classical2 | logreg | istella | accuracy (higher is better) | - | 0.924540 | - | cuml-gpu 0.924550 |
| classical2 | logreg | istella | logloss (lower is better) | - | 0.181245 | - | cuml-gpu 0.181262 |
| classical2 | logreg | istella | nonfinite_proba_rows | - | 0 | - | cuml-gpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | - | 0.763340 | - | cuml-gpu 0.763340 |
| classical2 | logreg | taxi | logloss (lower is better) | - | 0.538984 | - | cuml-gpu 0.538984 |
| classical2 | logreg | taxi | nonfinite_proba_rows | - | 0 | - | cuml-gpu 0 |
| classical2 | nystroem | istella | kernel_rel_error (lower is better) | - | - | - | sklearn-cpu 0.038958 |
| classical2 | nystroem | taxi | kernel_rel_error (lower is better) | - | - | - | sklearn-cpu 0.044369 |
| classical2 | rbf-sampler | istella | kernel_rel_error (lower is better) | - | 0.141980 | - | sklearn-cpu 0.137405 |
| classical2 | rbf-sampler | taxi | kernel_rel_error (lower is better) | - | 0.108549 | - | sklearn-cpu 0.083775 |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328682 | - | cuml-gpu -0.251259 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684423 | - | cuml-gpu 0.934403 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | - | cuml-gpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805042 | - | cuml-gpu 4.805051 |
| classical2 | spectral-embedding | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.799386 | - | cuml-gpu 0.882904; sklearn-cpu 0.461447 |
| classical2 | spectral-embedding | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.884889 | - | cuml-gpu 0.891595; sklearn-cpu 0.898012 |
| classical2 | spectral | istella | n_clusters | - | 8 | - | cuml-gpu 8; sklearn-cpu 8 |
| classical2 | spectral | istella | silhouette (higher is better) | - | 0.147668 | - | cuml-gpu 0.147757; sklearn-cpu 0.147699 |
| classical2 | spectral | istella | ari_vs_ours (1 is our partition exactly) | - | - | - | cuml-gpu 0.999106; sklearn-cpu 0.999826 |
| classical2 | spectral | taxi | n_clusters | - | 8 | - | cuml-gpu 8; sklearn-cpu 8 |
| classical2 | spectral | taxi | silhouette (higher is better) | - | 0.039910 | - | cuml-gpu 0.087609; sklearn-cpu 0.089894 |
| classical2 | spectral | taxi | ari_vs_ours (1 is our partition exactly) | - | - | - | cuml-gpu 0.574097; sklearn-cpu 0.582313 |
| classical2 | svr | istella | r2 (higher is better) | - | 0.318258 | - | cuml-gpu 0.318232 |
| classical2 | svr | istella | rmse (lower is better) | - | 0.680816 | - | cuml-gpu 0.680829 |
| classical2 | svr | taxi | r2 (higher is better) | - | 0.767551 | - | cuml-gpu 0.767551 |
| classical2 | svr | taxi | rmse (lower is better) | - | 7.680395 | - | cuml-gpu 7.680405 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | - | 0.999992 | - | cuml-gpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | - | 0.002554 | - | cuml-gpu 0.0001472 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999965 | - | cuml-gpu 0.999964 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | - | 0.003257 | - | cuml-gpu 0.003257 |
| classical2 | umap | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.979906 | - | cuml-gpu 0.979912 |
| classical2 | umap | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.990480 | - | cuml-gpu 0.992305 |
| neural | gemm-bf16 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 1.155e-07 | - | torch-eager-bf16 0.002764; torch-compile-bf16 0.002764 |
| neural | gemm-bf16 | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 1.000977; torch-compile-bf16 1.000977 |
| neural | gemm-bf16 | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 0.002764; torch-compile-bf16 0.002764 |
| neural | gemm-int8 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 0.000000 | - | torch-eager-int8 0.000000; torch-compile-int8 0.000000 |
| neural | gemm-int8 | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-int8 0.000000; torch-compile-int8 0.000000 |
| neural | gemm-int8 | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-int8 0.000000; torch-compile-int8 0.000000 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | - | torch-eager-fp32 1.401e-06; torch-eager-tf32 0.0002784; torch-compile-fp32 1.401e-06; torch-compile-tf32 0.0002784; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 0.0005341; torch-eager-tf32 0.100822; torch-compile-fp32 0.0005341; torch-compile-tf32 0.100822; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.475e-06; torch-eager-tf32 0.0002785; torch-compile-fp32 1.475e-06; torch-compile-tf32 0.0002785; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | - | torch-eager-fp32 9.018733; torch-eager-tf32 9.018733; torch-compile-fp32 9.018733; torch-compile-tf32 9.018732; torch-eager-bf16 9.018664; torch-compile-bf16 9.018669 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.371e-06; torch-eager-tf32 0.0007233; torch-compile-fp32 1.341e-06; torch-compile-tf32 0.0007184; torch-eager-bf16 0.006870; torch-compile-bf16 0.006304 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.882e-06; torch-eager-tf32 0.0009929; torch-compile-fp32 1.841e-06; torch-compile-tf32 0.0009862; torch-eager-bf16 0.009431; torch-compile-bf16 0.008653 |
| neural | lm-host-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.017858 | - | torch-cpu-eager-fp32 9.017857; torch-cpu-compile-fp32 9.017857; torch-cpu-eager-bf16 9.017747; torch-cpu-compile-bf16 9.017767 |
| neural | lm-host-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.367768 | - | torch-cpu-eager-fp32 8.367767; torch-cpu-compile-fp32 8.367766; torch-cpu-eager-bf16 8.368348; torch-cpu-compile-bf16 8.367754 |
| neural | lm-host-train-step | bytes | steps | - | 2 | - | torch-cpu-eager-fp32 2; torch-cpu-compile-fp32 2; torch-cpu-eager-bf16 2; torch-cpu-compile-bf16 2 |
| neural | lm-host-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-eager-fp32 9.537e-07; torch-cpu-compile-fp32 9.537e-07; torch-cpu-eager-bf16 0.0001106; torch-cpu-compile-bf16 9.06e-05 |
| neural | lm-host-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-eager-fp32 9.537e-07; torch-cpu-compile-fp32 1.907e-06; torch-cpu-eager-bf16 0.0005798; torch-cpu-compile-bf16 1.431e-05 |
| neural | lm-infer | bytes | mean_nll (lower is better) | - | 9.017857 | - | torch-cpu-eager-fp32 9.017857; torch-cpu-compile-fp32 9.017857; torch-cpu-eager-bf16 9.017748; torch-cpu-compile-bf16 9.017761 |
| neural | lm-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 7.227e-07; torch-cpu-compile-fp32 6.706e-07; torch-cpu-eager-bf16 0.006180; torch-cpu-compile-bf16 0.006585 |
| neural | lm-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 9.921e-07; torch-cpu-compile-fp32 9.205e-07; torch-cpu-eager-bf16 0.008484; torch-cpu-compile-bf16 0.009040 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | - | torch-eager-fp32 9.018733; torch-eager-tf32 9.018732; torch-compile-fp32 9.018734; torch-compile-tf32 9.018732; torch-eager-bf16 9.018402; torch-compile-bf16 9.018669 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.422411 | - | torch-eager-fp32 8.422415; torch-eager-tf32 8.422506; torch-compile-fp32 8.422415; torch-compile-tf32 8.422516; torch-eager-bf16 8.421570; torch-compile-bf16 8.420250 |
| neural | lm-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | lm-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 0.000000; torch-eager-tf32 9.537e-07; torch-compile-fp32 9.537e-07; torch-compile-tf32 9.537e-07; torch-eager-bf16 0.0003309; torch-compile-bf16 6.39e-05 |
| neural | lm-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 3.815e-06; torch-eager-tf32 9.537e-05; torch-compile-fp32 3.815e-06; torch-compile-tf32 0.0001049; torch-eager-bf16 0.0008411; torch-compile-bf16 0.002161 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.192e-07; torch-eager-tf32 3.815e-06; torch-eager-bf16 5.15e-05 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 5.945e-08; torch-eager-tf32 1.902e-06; torch-eager-bf16 2.568e-05 |
| neural | mamba1-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-07; torch-cpu-eager-bf16 5.15e-05 |
| neural | mamba1-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 5.948e-08; torch-cpu-eager-bf16 2.57e-05 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.146e-06; torch-eager-tf32 0.000509; torch-compile-fp32 2.146e-06; torch-compile-tf32 0.000509; torch-eager-bf16 0.007544; torch-compile-bf16 0.007544 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 7.295e-07; torch-eager-tf32 0.0001731; torch-compile-fp32 7.295e-07; torch-compile-tf32 0.0001731; torch-eager-bf16 0.002565; torch-compile-bf16 0.002565 |
| neural | mamba2-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 7.749e-07; torch-cpu-compile-fp32 7.749e-07; torch-cpu-eager-bf16 0.005920; torch-cpu-compile-bf16 0.005920 |
| neural | mamba2-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.751e-07; torch-cpu-compile-fp32 2.751e-07; torch-cpu-eager-bf16 0.002102; torch-cpu-compile-bf16 0.002102 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-eager-tf32 0.0001827; torch-compile-fp32 4.768e-07; torch-compile-tf32 0.0001798; torch-eager-bf16 0.002095; torch-compile-bf16 0.001982 |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.086e-07; torch-eager-tf32 7.993e-05; torch-compile-fp32 2.086e-07; torch-compile-tf32 7.867e-05; torch-eager-bf16 0.0009164; torch-compile-bf16 0.0008672 |
| neural | mamba3-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.384e-07; torch-cpu-compile-fp32 2.98e-07; torch-cpu-eager-bf16 0.002148; torch-cpu-compile-bf16 0.002040 |
| neural | mamba3-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.06e-07; torch-cpu-compile-fp32 1.324e-07; torch-cpu-eager-bf16 0.0009547; torch-cpu-compile-bf16 0.0009065 |
| neural | mlp-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-07; torch-cpu-compile-fp32 1.192e-07; torch-cpu-eager-bf16 0.004285; torch-cpu-compile-bf16 0.004285 |
| neural | mlp-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.1e-07; torch-cpu-compile-fp32 1.1e-07; torch-cpu-eager-bf16 0.003956; torch-cpu-compile-bf16 0.003956 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | - | torch-eager-fp32 1.160401; torch-eager-tf32 1.160392; torch-compile-fp32 1.160401; torch-compile-tf32 1.160392; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | - | torch-eager-fp32 1.123361; torch-eager-tf32 1.123355; torch-compile-fp32 1.123361; torch-compile-tf32 1.123355; torch-eager-bf16 1.123461; torch-compile-bf16 1.123462 |
| neural | mlp-train-step | gaussian | steps | - | 2 | - | torch-eager-fp32 2; torch-eager-tf32 2; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-eager-tf32 8.941e-06; torch-compile-fp32 2.384e-07; torch-compile-tf32 8.941e-06; torch-eager-bf16 9.656e-05; torch-compile-bf16 9.656e-05 |
| neural | mlp-train-step | gaussian | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-eager-tf32 5.484e-06; torch-compile-fp32 2.384e-07; torch-compile-tf32 5.603e-06; torch-eager-bf16 0.0001006; torch-compile-bf16 0.0001007 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-eager-fp32 5.635910; torch-eager-tf32 5.635948; torch-compile-fp32 5.635910; torch-compile-tf32 5.635956; torch-eager-bf16 5.635952; torch-compile-bf16 5.635985 |
| neural | samba-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.235e-06; torch-eager-tf32 0.001577; torch-compile-fp32 2.056e-06; torch-compile-tf32 0.001510; torch-eager-bf16 0.019621; torch-compile-bf16 0.016407 |
| neural | samba-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.257e-06; torch-eager-tf32 0.0008869; torch-compile-fp32 1.156e-06; torch-compile-tf32 0.0008487; torch-eager-bf16 0.011031; torch-compile-bf16 0.009224 |
| neural | samba-infer | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-cpu-eager-fp32 5.635910; torch-cpu-compile-fp32 5.635910; torch-cpu-eager-bf16 5.635976; torch-cpu-compile-bf16 5.635914 |
| neural | samba-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.176e-06; torch-cpu-compile-fp32 1.848e-06; torch-cpu-eager-bf16 0.018279; torch-cpu-compile-bf16 0.016166 |
| neural | samba-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.223e-06; torch-cpu-compile-fp32 1.039e-06; torch-cpu-eager-bf16 0.010276; torch-cpu-compile-bf16 0.009088 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | - | torch-compile-bf16 -; torch-compile-fp32 5.635910; torch-compile-tf32 5.635948; torch-eager-bf16 5.635952; torch-eager-fp32 5.635910; torch-eager-tf32 5.635948 |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | - | torch-compile-bf16 -; torch-compile-fp32 4.833934; torch-compile-tf32 4.833748; torch-eager-bf16 4.833967; torch-eager-fp32 4.833934; torch-eager-tf32 4.833591 |
| neural | samba-train-step | bytes | steps | - | 2 | - | torch-compile-bf16 -; torch-compile-fp32 2; torch-compile-tf32 2; torch-eager-bf16 2; torch-eager-fp32 2; torch-eager-tf32 2 |
| neural | samba-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-compile-bf16 -; torch-compile-fp32 0.000000; torch-compile-tf32 3.815e-05; torch-eager-bf16 4.196e-05; torch-eager-fp32 4.768e-07; torch-eager-tf32 3.815e-05 |
| neural | samba-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-compile-bf16 -; torch-compile-fp32 4.768e-07; torch-compile-tf32 0.0001855; torch-eager-bf16 3.338e-05; torch-eager-fp32 4.768e-07; torch-eager-tf32 0.0003433 |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-eager-tf32 0.0002124; torch-compile-fp32 4.768e-07; torch-compile-tf32 0.0002124; torch-eager-bf16 0.001819; torch-compile-bf16 0.001819 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.02e-07; torch-eager-tf32 4.545e-05; torch-compile-fp32 1.02e-07; torch-compile-tf32 4.545e-05; torch-eager-bf16 0.0003893; torch-compile-bf16 0.0003893 |
| neural | transformer-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 4.768e-07; torch-cpu-compile-fp32 9.537e-07; torch-cpu-eager-bf16 0.001941; torch-cpu-compile-bf16 0.001819 |
| neural | transformer-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.076e-07; torch-cpu-compile-fp32 2.153e-07; torch-cpu-eager-bf16 0.0004383; torch-cpu-compile-bf16 0.0004107 |
| trees | et | istella | auc (higher is better) | - | 0.937987 | - | sklearn-et-cpu 0.937659 |
| trees | et | istella | logloss (lower is better) | - | 0.189989 | - | sklearn-et-cpu 0.190177 |
| trees | et | taxi | auc (higher is better) | - | 0.618907 | - | sklearn-et-cpu 0.619262 |
| trees | et | taxi | logloss (lower is better) | - | 0.526142 | - | sklearn-et-cpu 0.525951 |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | 0.630363 | - | catboost-gpu 0.630433; xgboost-gpu 0.631978; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | 0.528463 | - | catboost-gpu 0.528509; xgboost-gpu 0.528548; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.979129 | - | catboost-gpu 0.983291; xgboost-gpu 0.983622; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.188483 | - | catboost-gpu 0.155812; xgboost-gpu 0.149263; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.625417 | - | catboost-gpu 0.632248; xgboost-gpu 0.630969; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.530232 | - | catboost-gpu 0.528018; xgboost-gpu 0.528678; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983749 | - | catboost-gpu 0.983599; xgboost-gpu 0.983622; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.149364 | - | catboost-gpu 0.149896; xgboost-gpu 0.149263; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.631154 | - | catboost-gpu 0.632012; xgboost-gpu 0.630969; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528317 | - | catboost-gpu 0.528025; xgboost-gpu 0.528678; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.903294 | - | catboost-gpu 0.907846; xgboost-gpu 0.910140; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.281958 | - | catboost-gpu 0.258180; xgboost-gpu 0.246803; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.596646 | - | catboost-gpu 0.599270; xgboost-gpu 0.601200; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.022664 | - | catboost-gpu 1.012704; xgboost-gpu 1.005204; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.972482 | - | catboost-gpu 0.979541; catboost-cpu - |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.227811 | - | catboost-gpu 0.190146; catboost-cpu - |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.620358 | - | catboost-gpu 0.629073; catboost-cpu - |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.531519 | - | catboost-gpu 0.528987; catboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.844605 | - | catboost-gpu 0.853929; xgboost-gpu 0.872796; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.711712 | - | catboost-gpu 0.719621; xgboost-gpu 0.738397; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.641863 | - | catboost-gpu 0.650025; xgboost-gpu 0.670093; catboost-cpu -; xgboost-cpu - |
| trees | gbdt-rank-yetirank | istella | map (higher is better) | - | 0.814902 | - | catboost-gpu 0.814091; xgboost-gpu 0.842476; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-rank-yetirank | istella | ndcg10 (higher is better) | - | 0.680993 | - | catboost-gpu 0.681202; xgboost-gpu 0.725584; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-rank-yetirank | istella | ndcg5 (higher is better) | - | 0.615076 | - | catboost-gpu 0.614769; xgboost-gpu 0.663588; lightgbm-cuda -; catboost-cpu -; xgboost-cpu -; lightgbm-cpu - |
| trees | gbdt-symmetric-1000 | istella | auc (higher is better) | - | 0.977075 | - | catboost-gpu 0.982462; catboost-cpu - |
| trees | gbdt-symmetric-1000 | istella | logloss (lower is better) | - | 0.203751 | - | catboost-gpu 0.170387; catboost-cpu - |
| trees | gbdt-symmetric-1000 | taxi | auc (higher is better) | - | 0.621310 | - | catboost-gpu 0.631809; catboost-cpu - |
| trees | gbdt-symmetric-1000 | taxi | logloss (lower is better) | - | 0.531329 | - | catboost-gpu 0.528128; catboost-cpu - |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.977075 | - | catboost-gpu 0.980020; catboost-cpu - |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.203751 | - | catboost-gpu 0.187222; catboost-cpu - |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.621310 | - | catboost-gpu 0.630348; catboost-cpu - |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.531329 | - | catboost-gpu 0.528586; catboost-cpu - |
| trees | iforest | istella | auc (higher is better) | - | 0.830358 | - | cuml-iforest-gpu 0.830358 |
| trees | iforest | taxi | auc (higher is better) | - | 0.551846 | - | cuml-iforest-gpu 0.551846 |
| trees | rf | istella | auc (higher is better) | - | 0.945385 | - | cuml-rf-gpu 0.945348 |
| trees | rf | istella | logloss (lower is better) | - | 0.182017 | - | cuml-rf-gpu 0.182069 |
| trees | rf | taxi | auc (higher is better) | - | 0.617838 | - | cuml-rf-gpu 0.617836 |
| trees | rf | taxi | logloss (lower is better) | - | 0.525953 | - | cuml-rf-gpu 0.525954 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | ard | istella | Xq | - | - | 5.1 | - | - | - | sklearn-cpu 6.1 ms (IDENTICAL/arm 0.845) |
| algos | ard | taxi | Xq | - | - | 0.4 | - | - | - | sklearn-cpu 0.5 ms (IDENTICAL/arm 0.795) |
| algos | bayesian-ridge | istella | Xq | - | - | 5.5 | - | - | - | sklearn-cpu 5.9 ms (IDENTICAL/arm 0.940) |
| algos | bayesian-ridge | taxi | Xq | - | - | 1.0 | - | - | - | sklearn-cpu 2.4 ms (IDENTICAL/arm 0.416) |
| algos | enet-cv | istella | Xq | - | - | 5.5 | - | - | - | sklearn-cpu 24.5 ms (IDENTICAL/arm 0.224) |
| algos | enet-cv | taxi | Xq | - | - | 1.3 | - | - | - | sklearn-cpu 0.8 ms (IDENTICAL/arm 1.615) |
| algos | gamma | istella | Xq | - | - | 5.3 | - | - | - | sklearn-cpu 33.2 ms (IDENTICAL/arm 0.159) |
| algos | gamma | taxi | Xq | - | - | 0.5 | - | - | - | sklearn-cpu 4.2 ms (IDENTICAL/arm 0.112) |
| algos | huber | istella | Xq | - | - | 5.2 | - | - | - | sklearn-cpu 41.6 ms (IDENTICAL/arm 0.125) |
| algos | huber | taxi | Xq | - | - | 1.4 | - | - | - | sklearn-cpu 20.3 ms (IDENTICAL/arm 0.070) |
| algos | lars | istella | Xq | - | - | 5.5 | - | - | - | sklearn-cpu 4.9 ms (IDENTICAL/arm 1.109); cuml-gpu 11.5 ms (IDENTICAL/arm 0.474) |
| algos | lars | taxi | Xq | - | - | 0.5 | - | - | - | sklearn-cpu 0.5 ms (IDENTICAL/arm 0.902); cuml-gpu 0.6 ms (IDENTICAL/arm 0.776) |
| algos | lasso-cv | istella | Xq | - | - | 5.4 | - | - | - | sklearn-cpu 8.9 ms (IDENTICAL/arm 0.609) |
| algos | lasso-cv | taxi | Xq | - | - | 1.3 | - | - | - | sklearn-cpu 3.7 ms (IDENTICAL/arm 0.345) |
| algos | lasso-lars | istella | Xq | - | - | 5.4 | - | - | - | sklearn-cpu 4.7 ms (IDENTICAL/arm 1.155) |
| algos | lasso-lars | taxi | Xq | - | - | 0.5 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 1.050) |
| algos | pa-clf | istella | Xq | - | - | 16.4 | - | - | - | sklearn-cpu 6.1 ms (IDENTICAL/arm 2.692) |
| algos | pa-clf | taxi | Xq | - | - | 13.4 | - | - | - | sklearn-cpu 1.8 ms (IDENTICAL/arm 7.339) |
| algos | pa-reg | istella | Xq | - | - | 5.4 | - | - | - | sklearn-cpu 5.8 ms (IDENTICAL/arm 0.932) |
| algos | pa-reg | taxi | Xq | - | - | 1.6 | - | - | - | sklearn-cpu 2.2 ms (IDENTICAL/arm 0.741) |
| algos | perceptron | istella | Xq | - | - | 16.6 | - | - | - | sklearn-cpu 6.9 ms (IDENTICAL/arm 2.419) |
| algos | perceptron | taxi | Xq | - | - | 14.5 | - | - | - | sklearn-cpu 0.9 ms (IDENTICAL/arm 16.445) |
| algos | poisson | istella | Xq | - | - | 5.4 | - | - | - | sklearn-cpu 33.1 ms (IDENTICAL/arm 0.163) |
| algos | poisson | taxi | Xq | - | - | 1.3 | - | - | - | sklearn-cpu 3.4 ms (IDENTICAL/arm 0.391) |
| algos | quantile | istella | Xq | - | - | 5.1 | - | - | - | sklearn-cpu 21.7 ms (IDENTICAL/arm 0.237) |
| algos | quantile | taxi | Xq | - | - | 1.6 | - | - | - | sklearn-cpu 4.3 ms (IDENTICAL/arm 0.373) |
| algos | ridge-clf | istella | Xq | - | - | 18.0 | - | - | - | sklearn-cpu 6.7 ms (IDENTICAL/arm 2.700) |
| algos | ridge-clf | taxi | Xq | - | - | 13.0 | - | - | - | sklearn-cpu 0.7 ms (IDENTICAL/arm 17.751) |
| algos | ridge-cv | istella | Xq | - | - | - | - | - | - | sklearn-cpu 6.0 ms (IDENTICAL/arm -) |
| algos | ridge-cv | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm -) |
| algos | sgd-clf | istella | Xq | - | - | 18.4 | - | - | - | sklearn-cpu 6.7 ms (IDENTICAL/arm 2.736); cuml-gpu 2.8 ms (IDENTICAL/arm 6.628) |
| algos | sgd-clf | taxi | Xq | - | - | 12.7 | - | - | - | sklearn-cpu 2.6 ms (IDENTICAL/arm 4.904); cuml-gpu 0.8 ms (IDENTICAL/arm 16.167) |
| algos | sgd-ocsvm | istella | Xq | - | - | 13.0 | - | - | - | sklearn-cpu 5.4 ms (IDENTICAL/arm 2.429) |
| algos | sgd-ocsvm | taxi | Xq | - | - | 9.4 | - | - | - | sklearn-cpu 0.9 ms (IDENTICAL/arm 10.672) |
| algos | sgd-reg | istella | Xq | - | - | 6.0 | - | - | - | sklearn-cpu 7.5 ms (IDENTICAL/arm 0.796); cuml-gpu 2.4 ms (IDENTICAL/arm 2.475) |
| algos | sgd-reg | taxi | Xq | - | - | 1.6 | - | - | - | sklearn-cpu 2.8 ms (IDENTICAL/arm 0.580); cuml-gpu 0.6 ms (IDENTICAL/arm 2.931) |
| algos | tweedie | istella | Xq | - | - | 5.1 | - | - | - | sklearn-cpu 29.6 ms (IDENTICAL/arm 0.174) |
| algos | tweedie | taxi | Xq | - | - | 0.5 | - | - | - | sklearn-cpu 3.3 ms (IDENTICAL/arm 0.145) |
| classical | kmeans | istella | Xq | 500000 | - | 80.0 | - | - | - | cuml-gpu 6.3 ms (IDENTICAL/arm 12.630); torch-gpu 1.3 ms (IDENTICAL/arm 62.616) |
| classical | kmeans | taxi | Xq | 500000 | - | 11.0 | - | - | - | cuml-gpu 2.8 ms (IDENTICAL/arm 3.895); torch-gpu 0.5 ms (IDENTICAL/arm 20.109) |
| classical | ols | istella | Xq | 500000 | - | 29.5 | - | - | - | cuml-gpu 3.9 ms (IDENTICAL/arm 7.478); torch-gpu 1.0 ms (IDENTICAL/arm 31.003); torch-gpu-eigh 1.0 ms (IDENTICAL/arm 30.509) |
| classical | ols | taxi | Xq | 500000 | - | 2.7 | - | - | - | cuml-gpu 1.3 ms (IDENTICAL/arm 2.133); torch-gpu 0.4 ms (IDENTICAL/arm 6.863); torch-gpu-eigh 0.4 ms (IDENTICAL/arm 6.767) |
| classical | pca | istella | Xq | 500000 | - | 45.5 | - | - | - | torch-gpu 2.3 ms (IDENTICAL/arm 19.538); cuml-gpu 10.4 ms (IDENTICAL/arm 3.574) |
| classical | pca | taxi | Xq | 500000 | - | 22.8 | - | - | - | torch-gpu 0.6 ms (IDENTICAL/arm 39.243); cuml-gpu 1.8 ms (IDENTICAL/arm 8.171) |
| classical | svc | istella | Xq | 10000 | - | 18.1 | - | - | - | cuml-gpu 13.1 ms (IDENTICAL/arm 1.382) |
| classical | svc | taxi | Xq | 10000 | - | 5.2 | - | - | - | cuml-gpu 5.3 ms (IDENTICAL/arm 0.985) |
| trees | et | istella | test | 500000 | - | 50.8 | - | - | - | sklearn-et-cpu 257.1 ms (IDENTICAL/arm 0.198) |
| trees | et | istella | large | 1000000 | - | 105.2 | - | - | - | sklearn-et-cpu 526.9 ms (IDENTICAL/arm 0.200) |
| trees | et | taxi | test | 500000 | - | 9.9 | - | - | - | sklearn-et-cpu 144.8 ms (IDENTICAL/arm 0.069) |
| trees | et | taxi | large | 1000000 | - | 19.4 | - | - | - | sklearn-et-cpu 240.1 ms (IDENTICAL/arm 0.081) |
| trees | gbdt-categorical | taxi | test | 500000 | - | 171.6 | - | - | - | catboost-gpu - ms (IDENTICAL/arm -); xgboost-gpu 69.2 ms (IDENTICAL/arm 2.480); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-categorical | taxi | large | 1000000 | - | 309.8 | - | - | - | catboost-gpu - ms (IDENTICAL/arm -); xgboost-gpu 193.8 ms (IDENTICAL/arm 1.599); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-depthwise | istella | test | 500000 | - | 68.0 | - | - | - | catboost-gpu 256.1 ms (IDENTICAL/arm 0.265); xgboost-gpu 98.0 ms (IDENTICAL/arm 0.694) |
| trees | gbdt-depthwise | istella | large | 1000000 | - | 103.0 | - | - | - | catboost-gpu 430.8 ms (IDENTICAL/arm 0.239); xgboost-gpu 186.5 ms (IDENTICAL/arm 0.552) |
| trees | gbdt-depthwise | taxi | test | 500000 | - | 39.4 | - | - | - | catboost-gpu 178.3 ms (IDENTICAL/arm 0.221); xgboost-gpu 10.6 ms (IDENTICAL/arm 3.705) |
| trees | gbdt-depthwise | taxi | large | 1000000 | - | 46.4 | - | - | - | catboost-gpu 307.4 ms (IDENTICAL/arm 0.151); xgboost-gpu 18.8 ms (IDENTICAL/arm 2.464) |
| trees | gbdt-lossguide | istella | test | 500000 | - | 71.0 | - | - | - | catboost-gpu 279.3 ms (IDENTICAL/arm 0.254); xgboost-gpu 97.0 ms (IDENTICAL/arm 0.732); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | istella | large | 1000000 | - | 109.2 | - | - | - | catboost-gpu 428.5 ms (IDENTICAL/arm 0.255); xgboost-gpu 187.1 ms (IDENTICAL/arm 0.584); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | test | 500000 | - | 39.3 | - | - | - | catboost-gpu 175.7 ms (IDENTICAL/arm 0.223); xgboost-gpu 10.2 ms (IDENTICAL/arm 3.858); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-lossguide | taxi | large | 1000000 | - | 46.7 | - | - | - | catboost-gpu 420.0 ms (IDENTICAL/arm 0.111); xgboost-gpu 20.4 ms (IDENTICAL/arm 2.292); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | istella | test | 500000 | - | 46.9 | - | - | - | catboost-gpu 141.7 ms (IDENTICAL/arm 0.331); xgboost-gpu 511.4 ms (IDENTICAL/arm 0.092); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | istella | large | 1000000 | - | 86.1 | - | - | - | catboost-gpu 201.5 ms (IDENTICAL/arm 0.428); xgboost-gpu 997.4 ms (IDENTICAL/arm 0.086); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | taxi | test | 500000 | - | 15.1 | - | - | - | catboost-gpu 71.5 ms (IDENTICAL/arm 0.212); xgboost-gpu 32.3 ms (IDENTICAL/arm 0.468); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-multiclass | taxi | large | 1000000 | - | 25.7 | - | - | - | catboost-gpu 92.4 ms (IDENTICAL/arm 0.278); xgboost-gpu 59.1 ms (IDENTICAL/arm 0.435); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-ordered | istella | test | 500000 | - | 34.1 | - | - | - | catboost-gpu 736.0 ms (IDENTICAL/arm 0.046) |
| trees | gbdt-ordered | istella | large | 1000000 | - | 65.0 | - | - | - | catboost-gpu 1442.3 ms (IDENTICAL/arm 0.045) |
| trees | gbdt-ordered | taxi | test | 500000 | - | 4.9 | - | - | - | catboost-gpu 67.5 ms (IDENTICAL/arm 0.072) |
| trees | gbdt-ordered | taxi | large | 1000000 | - | 8.4 | - | - | - | catboost-gpu 149.2 ms (IDENTICAL/arm 0.056) |
| trees | gbdt-rank-pairlogit | istella | test | 681250 | - | 47.4 | - | - | - | catboost-gpu 77.0 ms (IDENTICAL/arm 0.615); xgboost-gpu 75.7 ms (IDENTICAL/arm 0.626) |
| trees | gbdt-rank-pairlogit | istella | large | 1000000 | - | 65.0 | - | - | - | catboost-gpu 104.5 ms (IDENTICAL/arm 0.622); xgboost-gpu 87.9 ms (IDENTICAL/arm 0.739) |
| trees | gbdt-rank-yetirank | istella | test | 681250 | - | 42.3 | - | - | - | catboost-gpu 69.5 ms (IDENTICAL/arm 0.608); xgboost-gpu 61.2 ms (IDENTICAL/arm 0.691); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-rank-yetirank | istella | large | 1000000 | - | 62.4 | - | - | - | catboost-gpu 72.1 ms (IDENTICAL/arm 0.866); xgboost-gpu 87.6 ms (IDENTICAL/arm 0.713); lightgbm-cuda - ms (IDENTICAL/arm -) |
| trees | gbdt-symmetric-1000 | istella | test | 500000 | - | 37.0 | - | - | - | catboost-gpu 778.8 ms (IDENTICAL/arm 0.048) |
| trees | gbdt-symmetric-1000 | istella | large | 1000000 | - | 66.8 | - | - | - | catboost-gpu 1545.6 ms (IDENTICAL/arm 0.043) |
| trees | gbdt-symmetric-1000 | taxi | test | 500000 | - | 10.1 | - | - | - | catboost-gpu 81.5 ms (IDENTICAL/arm 0.124) |
| trees | gbdt-symmetric-1000 | taxi | large | 1000000 | - | 18.7 | - | - | - | catboost-gpu 162.5 ms (IDENTICAL/arm 0.115) |
| trees | gbdt-symmetric | istella | test | 500000 | - | 36.5 | - | - | - | catboost-gpu 637.6 ms (IDENTICAL/arm 0.057) |
| trees | gbdt-symmetric | istella | large | 1000000 | - | 65.5 | - | - | - | catboost-gpu 1263.2 ms (IDENTICAL/arm 0.052) |
| trees | gbdt-symmetric | taxi | test | 500000 | - | 6.7 | - | - | - | catboost-gpu 71.3 ms (IDENTICAL/arm 0.094) |
| trees | gbdt-symmetric | taxi | large | 1000000 | - | 11.9 | - | - | - | catboost-gpu 143.5 ms (IDENTICAL/arm 0.083) |
| trees | iforest | istella | test | 500000 | - | 947.7 | - | - | - | cuml-iforest-gpu 78.0 ms (IDENTICAL/arm 12.145) |
| trees | iforest | istella | large | 1000000 | - | 2047.9 | - | - | - | cuml-iforest-gpu 161.9 ms (IDENTICAL/arm 12.653) |
| trees | iforest | taxi | test | 500000 | - | 166.3 | - | - | - | cuml-iforest-gpu 6.9 ms (IDENTICAL/arm 24.239) |
| trees | iforest | taxi | large | 1000000 | - | 245.8 | - | - | - | cuml-iforest-gpu 15.9 ms (IDENTICAL/arm 15.487) |
| trees | rf | istella | test | 500000 | - | 54.8 | - | - | - | cuml-rf-gpu 48.0 ms (IDENTICAL/arm 1.141) |
| trees | rf | istella | large | 1000000 | - | 108.3 | - | - | - | cuml-rf-gpu 100.3 ms (IDENTICAL/arm 1.080) |
| trees | rf | taxi | test | 500000 | - | 9.1 | - | - | - | cuml-rf-gpu 9.0 ms (IDENTICAL/arm 1.014) |
| trees | rf | taxi | large | 1000000 | - | 18.8 | - | - | - | cuml-rf-gpu 19.2 ms (IDENTICAL/arm 0.978) |

## Trees

### et / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/et.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3763.2 | 3763.2..3763.2 | 1 | - | - | - | 24132.4 | - | auc=0.937987, logloss=0.189989 | yes | COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 16539.7 | 16539.7..16539.7 | 1 | 0.228 | - | - | 24890.4 | - | auc=0.937659, logloss=0.190177 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-et-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=ours,sklearn-et-cpu leaves=ours:1029236,sklearn-et-cpu:1006311 spread=0.0223 verdict=COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-et-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | false | false |
| class_weight | null | null |
| criterion | "gini" | "gini" |
| max_depth | 16 | 16 |
| max_features | "sqrt" | "sqrt" |
| max_leaves | null | null |
| max_samples | null | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | 0.0 | 0.0 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |

accepted difference: sklearn-et-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-et-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: sklearn-et-cpu max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 50.8 | 50.8..50.8 | 1 | - | - | - | auc=0.937987, auc_matches_fit=True, logloss=0.189989, logloss_matches_fit=True | yes | COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 257.1 | 257.1..257.1 | 1 | 0.198 | - | - | auc=0.937659, auc_matches_fit=True, logloss=0.190177, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 105.2 | 105.2..105.2 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 526.9 | 526.9..526.9 | 1 | 0.200 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

### et / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/et.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2519.7 | 2519.7..2519.7 | 1 | - | - | - | 10351.4 | - | auc=0.618907, logloss=0.526142 | yes | COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 12501.5 | 12501.5..12501.5 | 1 | 0.202 | - | - | 13627.4 | - | auc=0.619262, logloss=0.525951 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-et-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=ours,sklearn-et-cpu leaves=ours:881399,sklearn-et-cpu:906974 spread=0.0282 verdict=COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-et-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | false | false |
| class_weight | null | null |
| criterion | "gini" | "gini" |
| max_depth | 16 | 16 |
| max_features | "sqrt" | "sqrt" |
| max_leaves | null | null |
| max_samples | null | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | 0.0 | 0.0 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |

accepted difference: sklearn-et-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-et-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: sklearn-et-cpu max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 9.9 | 9.9..9.9 | 1 | - | - | - | auc=0.618907, auc_matches_fit=True, logloss=0.526142, logloss_matches_fit=True | yes | COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 144.8 | 144.8..144.8 | 1 | 0.069 | - | - | auc=0.619262, auc_matches_fit=True, logloss=0.525951, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 19.4 | 19.4..19.4 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 240.1 | 240.1..240.1 | 1 | 0.081 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-categorical.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15306.3 | 15306.3..15306.3 | 1 | - | - | - | 5418.5 | 1524.0 | auc=0.630363, logloss=0.528463 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 26094.5 | 26094.5..26094.5 | 1 | 0.587 | - | - | 6472.1 | 1524.0 | auc=0.630433, logloss=0.528509 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 13560.4 | 13560.4..13560.4 | 1 | 1.129 | - | - | 4475.6 | 1524.0 | auc=0.631978, logloss=0.528548 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-categorical arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:98864,ours:42048,xgboost-gpu:102162 spread=0.5884 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | ours | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - |
| class_weight | - | null | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - |
| random_strength | 0.0 | - | 0.0 | - |
| reg_alpha | - | 0.0 | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "NewtonL2" | - | "NewtonL2" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 171.6 | 171.6..171.6 | 1 | - | - | - | auc=0.630363, auc_matches_fit=True, logloss=0.528463, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | REFUSED(CatBoostError during warm-up: catboost/libs/model/cuda/evaluator.cpp:22: Model contains categorical features, GPU evaluation impossible) |
| xgboost-gpu | test | 500000 | 69.2 | 69.2..69.2 | 1 | 2.480 | - | - | auc=0.631978, auc_matches_fit=True, logloss=0.528548, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 309.8 | 309.8..309.8 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | REFUSED(CatBoostError during warm-up: catboost/libs/model/cuda/evaluator.cpp:22: Model contains categorical features, GPU evaluation impossible) |
| xgboost-gpu | large | 1000000 | 193.8 | 193.8..193.8 | 1 | 1.599 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict_proba(X) on the float32 codes, column 1

inference call, catboost-gpu: catboost predict_proba(int64 categorical frame built in the clock, task_type GPU), column 1

inference call, xgboost-gpu: xgboost XGBClassifier.predict_proba(pandas CategoricalDtype frame built in the clock; inplace_predict takes no category frame here), column 1

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6410.0 | 6410.0..6410.0 | 1 | - | - | - | 8697.5 | 1270.0 | auc=0.979129, logloss=0.188483 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 9362.5 | 9362.5..9362.5 | 1 | 0.685 | - | - | 7027.2 | 1270.0 | auc=0.983291, logloss=0.155812 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 8075.5 | 8075.5..8075.5 | 1 | 0.794 | - | - | 8157.7 | 1270.0 | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:105636,ours:38550,xgboost-gpu:107148 spread=0.6402 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours | xgboost-gpu |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | "No" | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 |
| grow_policy | "Depthwise" | "Depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | - |
| leaf_estimation_method | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | - |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | null | 0.0 |
| min_samples_leaf | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | - |
| random_strength | 0.0 | 0.0 | - |
| reg_alpha | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 |
| subsample | null | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 68.0 | 68.0..68.0 | 1 | - | - | - | auc=0.979129, auc_matches_fit=True, logloss=0.188483, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 256.1 | 256.1..256.1 | 1 | 0.265 | - | - | auc=0.983291, auc_matches_fit=True, logloss=0.155812, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 98.0 | 98.0..98.0 | 1 | 0.694 | - | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 103.0 | 103.0..103.0 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 430.8 | 430.8..430.8 | 1 | 0.239 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 186.5 | 186.5..186.5 | 1 | 0.552 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4277.0 | 4277.0..4277.0 | 1 | - | - | - | 3083.6 | 1266.0 | auc=0.625417, logloss=0.530232 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 6485.3 | 6485.3..6485.3 | 1 | 0.659 | - | - | 2961.0 | 1266.0 | auc=0.632248, logloss=0.528018 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 3249.7 | 3249.7..3249.7 | 1 | 1.316 | - | - | 2821.7 | 1266.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:103978,ours:41324,xgboost-gpu:91234 spread=0.6026 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours | xgboost-gpu |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | "No" | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 |
| grow_policy | "Depthwise" | "Depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | - |
| leaf_estimation_method | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | - |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | null | 0.0 |
| min_samples_leaf | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | - |
| random_strength | 0.0 | 0.0 | - |
| reg_alpha | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 |
| subsample | null | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 39.4 | 39.4..39.4 | 1 | - | - | - | auc=0.625417, auc_matches_fit=True, logloss=0.530232, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 178.3 | 178.3..178.3 | 1 | 0.221 | - | - | auc=0.632248, auc_matches_fit=True, logloss=0.528018, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 10.6 | 10.6..10.6 | 1 | 3.705 | - | - | auc=0.630969, auc_matches_fit=True, logloss=0.528678, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 46.4 | 46.4..46.4 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 307.4 | 307.4..307.4 | 1 | 0.151 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 18.8 | 18.8..18.8 | 1 | 2.464 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20966.3 | 20966.3..20966.3 | 1 | - | - | - | 9314.8 | 1266.0 | auc=0.983749, logloss=0.149364 | yes | COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 22265.6 | 22265.6..22265.6 | 1 | 0.942 | - | - | 7621.2 | 1266.0 | auc=0.983599, logloss=0.149896 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 11997.7 | 11997.7..11997.7 | 1 | 1.748 | - | - | 8630.0 | 1266.0 | auc=0.983622, logloss=0.149263 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:116618,ours:113893,xgboost-gpu:107148 spread=0.0812 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | ours | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - |
| class_weight | - | null | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - |
| random_strength | 0.0 | - | 0.0 | - |
| reg_alpha | - | 0.0 | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "NewtonL2" | - | "NewtonL2" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 71.0 | 71.0..71.0 | 1 | - | - | - | auc=0.983749, auc_matches_fit=True, logloss=0.149364, logloss_matches_fit=True | yes | COMPARABLE | ok |
| catboost-gpu | test | 500000 | 279.3 | 279.3..279.3 | 1 | 0.254 | - | - | auc=0.983599, auc_matches_fit=True, logloss=0.149896, logloss_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 97.0 | 97.0..97.0 | 1 | 0.732 | - | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 109.2 | 109.2..109.2 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 428.5 | 428.5..428.5 | 1 | 0.255 | - | - | - | yes | COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 187.1 | 187.1..187.1 | 1 | 0.584 | - | - | - | yes | COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8046.6 | 8046.6..8046.6 | 1 | - | - | - | 3104.7 | 1266.0 | auc=0.631154, logloss=0.528317 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 15946.6 | 15946.6..15946.6 | 1 | 0.505 | - | - | 2967.1 | 1266.0 | auc=0.632012, logloss=0.528025 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 6550.6 | 6550.6..6550.6 | 1 | 1.228 | - | - | 2893.9 | 1266.0 | auc=0.630969, logloss=0.528678 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:103595,ours:56739,xgboost-gpu:91234 spread=0.4523 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | ours | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - |
| class_weight | - | null | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - |
| random_strength | 0.0 | - | 0.0 | - |
| reg_alpha | - | 0.0 | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "NewtonL2" | - | "NewtonL2" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 39.3 | 39.3..39.3 | 1 | - | - | - | auc=0.631154, auc_matches_fit=True, logloss=0.528317, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 175.7 | 175.7..175.7 | 1 | 0.223 | - | - | auc=0.632012, auc_matches_fit=True, logloss=0.528025, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 10.2 | 10.2..10.2 | 1 | 3.858 | - | - | auc=0.630969, auc_matches_fit=True, logloss=0.528678, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 46.7 | 46.7..46.7 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 420.0 | 420.0..420.0 | 1 | 0.111 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 20.4 | 20.4..20.4 | 1 | 2.292 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU, thread_count -1) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:25: Model is not oblivious, GPU evaluatio), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the rows uploaded and the probability copied back inside the clock

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10314.7 | 10314.7..10314.7 | 1 | - | - | - | 9323.7 | 1582.0 | accuracy=0.903294, mlogloss=0.281958 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 14789.2 | 14789.2..14789.2 | 1 | 0.697 | - | - | 7636.1 | 1582.0 | accuracy=0.907846, mlogloss=0.258180 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 32408.6 | 32408.6..32408.6 | 1 | 0.318 | - | - | 8690.1 | 1582.0 | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:128000,ours:128000,xgboost-gpu:88137 spread=0.3114 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | ours | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - |
| class_weight | - | null | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "MultiClass" | - | "MultiClass" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | null | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - |
| min_split_gain | - | 0.0 | null | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - |
| random_strength | 1.0 | - | 1.0 | - |
| reg_alpha | - | 0.0 | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 46.9 | 46.9..46.9 | 1 | - | - | - | accuracy=0.903294, accuracy_matches_fit=True, mlogloss=0.281958, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 141.7 | 141.7..141.7 | 1 | 0.331 | - | - | accuracy=0.907846, accuracy_matches_fit=True, mlogloss=0.258180, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 511.4 | 511.4..511.4 | 1 | 0.092 | - | - | accuracy=0.910140, accuracy_matches_fit=True, mlogloss=0.246803, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 86.1 | 86.1..86.1 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 201.5 | 201.5..201.5 | 1 | 0.428 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 997.4 | 997.4..997.4 | 1 | 0.086 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:28: Model is not one-dimensional, GPU eva), the (rows, n_classes) probability matrix

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the (rows, n_classes) probability matrix

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6633.4 | 6633.4..6633.4 | 1 | - | - | - | 3175.4 | 1378.0 | accuracy=0.596646, mlogloss=1.022664 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 10105.2 | 10105.2..10105.2 | 1 | 0.656 | - | - | 3052.7 | 1378.0 | accuracy=0.599270, mlogloss=1.012704 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 13072.6 | 13072.6..13072.6 | 1 | 0.507 | - | - | 3284.4 | 1378.0 | accuracy=0.601200, mlogloss=1.005204 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:128000,ours:128000,xgboost-gpu:99730 spread=0.2209 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | ours | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - |
| class_weight | - | null | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "MultiClass" | - | "MultiClass" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | null | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - |
| min_split_gain | - | 0.0 | null | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - |
| random_strength | 1.0 | - | 1.0 | - |
| reg_alpha | - | 0.0 | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 15.1 | 15.1..15.1 | 1 | - | - | - | accuracy=0.596646, accuracy_matches_fit=True, mlogloss=1.022664, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 71.5 | 71.5..71.5 | 1 | 0.212 | - | - | accuracy=0.599270, accuracy_matches_fit=True, mlogloss=1.012704, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | 32.3 | 32.3..32.3 | 1 | 0.468 | - | - | accuracy=0.601200, accuracy_matches_fit=True, mlogloss=1.005204, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 25.7 | 25.7..25.7 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 92.4 | 92.4..92.4 | 1 | 0.278 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 59.1 | 59.1..59.1 | 1 | 0.435 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | - | NOT-COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, catboost-gpu: catboost predict_proba(X, task_type CPU) (task_type GPU refused: catboost/libs/model/cuda/evaluator.cpp:28: Model is not one-dimensional, GPU eva), the (rows, n_classes) probability matrix

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, the (rows, n_classes) probability matrix

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-ordered.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 143456.7 | 143456.7..143456.7 | 1 | - | - | - | 8001.8 | 4782.0 | auc=0.972482, logloss=0.227811 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 38076.0 | 38076.0..38076.0 | 1 | 3.768 | - | - | 6338.1 | 4782.0 | auc=0.979541, logloss=0.190146 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-ordered arms=catboost-gpu,ours leaves=catboost-gpu:128000,ours:31748 spread=0.7520 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours |
|---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) |
| boosting_type | "Ordered" | "Ordered" |
| bootstrap_type | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" |
| max_bin | 255 | 255 |
| max_depth | 8 | 8 |
| max_leaves | 256 | 256 |
| min_child_weight | - | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | - | null |
| n_estimators | 500 | 500 |
| nan_mode | "Min" | "Min" |
| random_strength | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" |
| seed | 7 | 7 |
| subsample | null | null |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 34.1 | 34.1..34.1 | 1 | - | - | - | auc=0.972482, auc_matches_fit=True, logloss=0.227811, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 736.0 | 736.0..736.0 | 1 | 0.046 | - | - | auc=0.979541, auc_matches_fit=True, logloss=0.190146, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 65.0 | 65.0..65.0 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 1442.3 | 1442.3..1442.3 | 1 | 0.045 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type GPU, thread_count -1), column 1

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-ordered.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 244419.0 | 244419.0..244419.0 | 1 | - | - | - | 2764.5 | 7854.0 | auc=0.620358, logloss=0.531519 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 20726.1 | 20726.1..20726.1 | 1 | 11.793 | - | - | 2633.1 | 7854.0 | auc=0.629073, logloss=0.528987 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-ordered arms=catboost-gpu,ours leaves=catboost-gpu:96372,ours:17836 spread=0.8149 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours |
|---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) |
| boosting_type | "Ordered" | "Ordered" |
| bootstrap_type | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" |
| max_bin | 255 | 255 |
| max_depth | 8 | 8 |
| max_leaves | 256 | 256 |
| min_child_weight | - | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | - | null |
| n_estimators | 500 | 500 |
| nan_mode | "Min" | "Min" |
| random_strength | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" |
| seed | 7 | 7 |
| subsample | null | null |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 4.9 | 4.9..4.9 | 1 | - | - | - | auc=0.620358, auc_matches_fit=True, logloss=0.531519, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 67.5 | 67.5..67.5 | 1 | 0.072 | - | - | auc=0.629073, auc_matches_fit=True, logloss=0.528987, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 8.4 | 8.4..8.4 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 149.2 | 149.2..149.2 | 1 | 0.056 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type GPU, thread_count -1), column 1

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-pairlogit.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4278.4 | 4278.4..4278.4 | 1 | - | - | - | 10010.0 | 1836.0 | map=0.844605, ndcg10=0.711712, ndcg5=0.641863 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 3806.7 | 3806.7..3806.7 | 1 | 1.124 | - | - | 8509.7 | 1836.0 | map=0.853929, ndcg10=0.719621, ndcg5=0.650025 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 3806.7 | 3806.7..3806.7 | 1 | 1.124 | - | - | 9099.1 | 1836.0 | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu, xgboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-rank-pairlogit arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:6400,ours:4332,xgboost-gpu:6395 spread=0.3231 verdict=NOT-COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours | xgboost-gpu |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | "No" | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | - |
| leaf_estimation_method | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "PairLogit" | "PairLogit" | - |
| max_bin | 255 | 255 | 255 |
| max_depth | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 |
| min_child_weight | - | null | 0.0 |
| min_samples_leaf | 1 | 1 | - |
| min_split_gain | - | null | 0.0 |
| n_estimators | 100 | 100 | 100 |
| nan_mode | "Min" | "Min" | - |
| random_strength | 1.0 | 1.0 | - |
| reg_alpha | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | 1.0 | 1.0 |
| score_function | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 |
| subsample | null | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 681250 | 47.4 | 47.4..47.4 | 1 | - | - | - | map=0.844605, map_matches_fit=True, ndcg10=0.711712, ndcg10_matches_fit=True, ndcg5=0.641863, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 681250 | 77.0 | 77.0..77.0 | 1 | 0.615 | - | - | map=0.853929, map_matches_fit=True, ndcg10=0.719621, ndcg10_matches_fit=True, ndcg5=0.650025, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 681250 | 75.7 | 75.7..75.7 | 1 | 0.626 | - | - | map=0.872796, map_matches_fit=True, ndcg10=0.738397, ndcg10_matches_fit=True, ndcg5=0.670093, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 65.0 | 65.0..65.0 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 104.5 | 104.5..104.5 | 1 | 0.622 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 87.9 | 87.9..87.9 | 1 | 0.739 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, catboost-gpu: catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the ranker: a host apply on every box), raw ranking scores

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, raw ranking scores

### gbdt-rank-yetirank / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-yetirank.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2602.8 | 2602.8..2602.8 | 1 | - | - | - | 9378.6 | 1324.0 | map=0.814902, ndcg10=0.680993, ndcg5=0.615076 | yes | COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 2562.0 | 2562.0..2562.0 | 1 | 1.016 | - | - | 7719.8 | 1324.0 | map=0.814091, ndcg10=0.681202, ndcg5=0.614769 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | 3984.9 | 3984.9..3984.9 | 1 | 0.653 | - | - | 8660.5 | 1324.0 | map=0.842476, ndcg10=0.725584, ndcg5=0.663588 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cuda | lightgbm | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option) (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: xgboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against ) (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, lightgbm-cuda, catboost-cpu, xgboost-cpu, lightgbm-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-rank-yetirank arms=catboost-gpu,ours,xgboost-gpu leaves=catboost-gpu:6400,ours:6400,xgboost-gpu:6399 spread=0.0002 verdict=COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | lightgbm-cuda | ours | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - |
| class_weight | - | null | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "YetiRank" | - | "YetiRank" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 |
| min_child_weight | - | 0.001 | null | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - |
| min_split_gain | - | 0.0 | null | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | - | "Min" | - |
| random_strength | 1.0 | - | 1.0 | - |
| reg_alpha | - | 0.0 | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cuda boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cuda min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cuda min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cuda min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cuda subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 681250 | 42.3 | 42.3..42.3 | 1 | - | - | - | map=0.814902, map_matches_fit=True, ndcg10=0.680993, ndcg10_matches_fit=True, ndcg5=0.615076, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| catboost-gpu | test | 681250 | 69.5 | 69.5..69.5 | 1 | 0.608 | - | - | map=0.814091, map_matches_fit=True, ndcg10=0.681202, ndcg10_matches_fit=True, ndcg5=0.614769, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-gpu | test | 681250 | 61.2 | 61.2..61.2 | 1 | 0.691 | - | - | map=0.842476, map_matches_fit=True, ndcg10=0.725584, ndcg10_matches_fit=True, ndcg5=0.663588, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cuda | test | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |
| mojolearn IDENTICAL | large | 1000000 | 62.4 | 62.4..62.4 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 72.1 | 72.1..72.1 | 1 | 0.866 | - | - | - | yes | COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | 87.6 | 87.6..87.6 | 1 | 0.713 | - | - | - | yes | COMPARABLE | ok |
| lightgbm-cuda | large | - | - | - | 0 | - | - | - | - | - | COMPARABLE | UNKNOWN(no inference lines) |

inference call, ours: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, catboost-gpu: catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the ranker: a host apply on every box), raw ranking scores

inference call, xgboost-gpu: xgboost Booster.inplace_predict(cupy.asarray(X)) then cupy.asnumpy, raw ranking scores

### gbdt-symmetric-1000 / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7657.3 | 7657.3..7657.3 | 1 | - | - | - | 7978.2 | 1198.0 | auc=0.977075, logloss=0.203751 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 13830.7 | 13830.7..13830.7 | 1 | 0.554 | - | - | 6341.5 | 1198.0 | auc=0.982462, logloss=0.170387 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-gpu,ours leaves=catboost-gpu:256000,ours:67958 spread=0.7345 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, 1000 trees (CatBoost's own default iteration count; see CATBOOST_DEFAULTS) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours |
|---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" |
| max_bin | 255 | 255 |
| max_depth | 8 | 8 |
| max_leaves | 256 | 256 |
| min_child_weight | - | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | - | null |
| n_estimators | 1000 | 1000 |
| nan_mode | "Min" | "Min" |
| random_strength | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" |
| seed | 7 | 7 |
| subsample | null | null |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 37.0 | 37.0..37.0 | 1 | - | - | - | auc=0.977075, auc_matches_fit=True, logloss=0.203751, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 778.8 | 778.8..778.8 | 1 | 0.048 | - | - | auc=0.982462, auc_matches_fit=True, logloss=0.170387, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 66.8 | 66.8..66.8 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 1545.6 | 1545.6..1545.6 | 1 | 0.043 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type GPU, thread_count -1), column 1

### gbdt-symmetric-1000 / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5850.8 | 5850.8..5850.8 | 1 | - | - | - | 2616.6 | 940.0 | auc=0.621310, logloss=0.531329 | yes | COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 10274.2 | 10274.2..10274.2 | 1 | 0.569 | - | - | 2482.6 | 940.0 | auc=0.631809, logloss=0.528128 | yes | COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-gpu,ours leaves=catboost-gpu:255408,ours:251900 spread=0.0137 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, 1000 trees (CatBoost's own default iteration count; see CATBOOST_DEFAULTS) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours |
|---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" |
| max_bin | 255 | 255 |
| max_depth | 8 | 8 |
| max_leaves | 256 | 256 |
| min_child_weight | - | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | - | null |
| n_estimators | 1000 | 1000 |
| nan_mode | "Min" | "Min" |
| random_strength | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" |
| seed | 7 | 7 |
| subsample | null | null |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 10.1 | 10.1..10.1 | 1 | - | - | - | auc=0.621310, auc_matches_fit=True, logloss=0.531329, logloss_matches_fit=True | yes | COMPARABLE | ok |
| catboost-gpu | test | 500000 | 81.5 | 81.5..81.5 | 1 | 0.124 | - | - | auc=0.631809, auc_matches_fit=True, logloss=0.528128, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 18.7 | 18.7..18.7 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 162.5 | 162.5..162.5 | 1 | 0.115 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type GPU, thread_count -1), column 1

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5257.9 | 5257.9..5257.9 | 1 | - | - | - | 7971.6 | 1198.0 | auc=0.977075, logloss=0.203751 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 7580.6 | 7580.6..7580.6 | 1 | 0.694 | - | - | 6320.6 | 1198.0 | auc=0.980020, logloss=0.187222 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | NOT-COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-gpu,ours leaves=catboost-gpu:128000,ours:66958 spread=0.4769 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours |
|---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" |
| max_bin | 255 | 255 |
| max_depth | 8 | 8 |
| max_leaves | 256 | 256 |
| min_child_weight | - | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | - | null |
| n_estimators | 500 | 500 |
| nan_mode | "Min" | "Min" |
| random_strength | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" |
| seed | 7 | 7 |
| subsample | null | null |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 36.5 | 36.5..36.5 | 1 | - | - | - | auc=0.977075, auc_matches_fit=True, logloss=0.203751, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-gpu | test | 500000 | 637.6 | 637.6..637.6 | 1 | 0.057 | - | - | auc=0.980020, auc_matches_fit=True, logloss=0.187222, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 65.5 | 65.5..65.5 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 1263.2 | 1263.2..1263.2 | 1 | 0.052 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type GPU, thread_count -1), column 1

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3141.1 | 3141.1..3141.1 | 1 | - | - | - | 2596.8 | 940.0 | auc=0.621310, logloss=0.531329 | yes | COMPARABLE | wheel | ok |
| catboost-gpu | catboost | gpu | opponent | 5721.3 | 5721.3..5721.3 | 1 | 0.549 | - | - | 2467.7 | 940.0 | auc=0.630348, logloss=0.528586 | yes | COMPARABLE | - | ok (measured this run) |
| catboost-cpu | catboost | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | COMPARABLE | - | REFUSED(GPU-PATH-ONLY: catboost-cpu is a CPU arm and cpu was not requested on this accelerator box. On NVIDIA we compare against) (measured this run) |

memory, ours, catboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

memory, catboost-cpu: host not sampled; GPU not sampled

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-gpu,ours leaves=catboost-gpu:127408,ours:123900 spread=0.0275 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-gpu | ours |
|---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" |
| max_bin | 255 | 255 |
| max_depth | 8 | 8 |
| max_leaves | 256 | 256 |
| min_child_weight | - | null |
| min_samples_leaf | 1 | 1 |
| min_split_gain | - | null |
| n_estimators | 500 | 500 |
| nan_mode | "Min" | "Min" |
| random_strength | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" |
| seed | 7 | 7 |
| subsample | null | null |

accepted difference: catboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 6.7 | 6.7..6.7 | 1 | - | - | - | auc=0.621310, auc_matches_fit=True, logloss=0.531329, logloss_matches_fit=True | yes | COMPARABLE | ok |
| catboost-gpu | test | 500000 | 71.3 | 71.3..71.3 | 1 | 0.094 | - | - | auc=0.630348, auc_matches_fit=True, logloss=0.528586, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 11.9 | 11.9..11.9 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| catboost-gpu | large | 1000000 | 143.5 | 143.5..143.5 | 1 | 0.083 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-gpu: catboost predict_proba(X, task_type GPU, thread_count -1), column 1

### iforest / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/iforest.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 502.8 | 502.8..502.8 | 1 | - | - | - | 21350.7 | 2222.0 | auc=0.830358 | yes | UNKNOWN | wheel | ok |
| cuml-iforest-gpu | cuml | gpu | opponent | 2575.1 | 2575.1..2575.1 | 1 | 0.195 | - | - | 21422.4 | 2222.0 | auc=0.830358 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, cuml-iforest-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

FSPEED-FIT-VERDICT: `lane=iforest arms=cuml-iforest-gpu,ours leaves=- spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-iforest-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| bootstrap | false | false |
| contamination | "auto" | "auto" |
| max_depth | null | null |
| max_features | 1.0 | 1.0 |
| max_samples | 256 | 256 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |

accepted difference: cuml-iforest-gpu max_depth: ours None is the auto depth ceil(log2(max_samples)) = 8; sklearn has no max_depth parameter and fixes the same value

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 947.7 | 947.7..947.7 | 1 | - | - | - | auc=0.830358, auc_matches_fit=True | yes | UNKNOWN | ok |
| cuml-iforest-gpu | test | 500000 | 78.0 | 78.0..78.0 | 1 | 12.145 | - | - | auc=0.830358, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 2047.9 | 2047.9..2047.9 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| cuml-iforest-gpu | large | 1000000 | 161.9 | 161.9..161.9 | 1 | 12.653 | - | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, cuml-iforest-gpu: cuml IsolationForest.score_samples(host X)

### iforest / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/iforest.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 158.3 | 158.3..158.3 | 1 | - | - | - | 2652.8 | 686.0 | auc=0.551846 | yes | UNKNOWN | wheel | ok |
| cuml-iforest-gpu | cuml | gpu | opponent | 178.6 | 178.6..178.6 | 1 | 0.886 | - | - | 2694.1 | 686.0 | auc=0.551846 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, cuml-iforest-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

FSPEED-FIT-VERDICT: `lane=iforest arms=cuml-iforest-gpu,ours leaves=- spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-iforest-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| bootstrap | false | false |
| contamination | "auto" | "auto" |
| max_depth | null | null |
| max_features | 1.0 | 1.0 |
| max_samples | 256 | 256 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |

accepted difference: cuml-iforest-gpu max_depth: ours None is the auto depth ceil(log2(max_samples)) = 8; sklearn has no max_depth parameter and fixes the same value

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 166.3 | 166.3..166.3 | 1 | - | - | - | auc=0.551846, auc_matches_fit=True | yes | UNKNOWN | ok |
| cuml-iforest-gpu | test | 500000 | 6.9 | 6.9..6.9 | 1 | 24.239 | - | - | auc=0.551846, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 245.8 | 245.8..245.8 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| cuml-iforest-gpu | large | 1000000 | 15.9 | 15.9..15.9 | 1 | 15.487 | - | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, cuml-iforest-gpu: cuml IsolationForest.score_samples(host X)

### rf / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/rf.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5045.3 | 5045.3..5045.3 | 1 | - | - | - | 21618.1 | 3256.0 | auc=0.945385, logloss=0.182017 | yes | UNKNOWN | wheel | ok |
| cuml-rf-gpu | cuml | gpu | opponent | 24051.6 | 24051.6..24051.6 | 1 | 0.210 | - | - | 21698.2 | 3260.0 | auc=0.945348, logloss=0.182069 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

FSPEED-FIT-VERDICT: `lane=rf arms=cuml-rf-gpu,ours leaves=ours:125692 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-rf-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| bootstrap | true | true |
| class_weight | null | null |
| criterion | - | "gini" |
| max_bin | 128 | 128 |
| max_depth | 8 | 8 |
| max_features | "sqrt" | "sqrt" |
| max_leaves | -1 | -1 |
| max_samples | 1.0 | 1.0 |
| min_samples_leaf | 1 | 1 |
| min_split_gain | 0.0 | 0.0 |
| n_estimators | 500 | 500 |
| seed | 7 | 7 |

accepted difference: cuml-rf-gpu class_weight: None on every arm is unit class weights, the value each library defines for None

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 54.8 | 54.8..54.8 | 1 | - | - | - | auc=0.945385, auc_matches_fit=True, logloss=0.182017, logloss_matches_fit=True | yes | UNKNOWN | ok |
| cuml-rf-gpu | test | 500000 | 48.0 | 48.0..48.0 | 1 | 1.141 | - | - | auc=0.945348, auc_matches_fit=True, logloss=0.182069, logloss_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 108.3 | 108.3..108.3 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| cuml-rf-gpu | large | 1000000 | 100.3 | 100.3..100.3 | 1 | 1.080 | - | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, cuml-rf-gpu: cuml RandomForest.predict_proba(host X) (FIL conversion refused: 'RandomForestClassifier' object has no attribute 'convert_to_fil_model')

### rf / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/rf.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5798.2 | 5798.2..5798.2 | 1 | - | - | - | 2993.5 | 1720.0 | auc=0.617838, logloss=0.525953 | yes | UNKNOWN | wheel | ok |
| cuml-rf-gpu | cuml | gpu | opponent | 21000.3 | 21000.3..21000.3 | 1 | 0.276 | - | - | 3056.4 | 1724.0 | auc=0.617836, logloss=0.525954 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, cuml-rf-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak) (the process total: every arm in this one process)

FSPEED-FIT-VERDICT: `lane=rf arms=cuml-rf-gpu,ours leaves=ours:123231 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-rf-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| bootstrap | true | true |
| class_weight | null | null |
| criterion | - | "gini" |
| max_bin | 128 | 128 |
| max_depth | 8 | 8 |
| max_features | "sqrt" | "sqrt" |
| max_leaves | -1 | -1 |
| max_samples | 1.0 | 1.0 |
| min_samples_leaf | 1 | 1 |
| min_split_gain | 0.0 | 0.0 |
| n_estimators | 500 | 500 |
| seed | 7 | 7 |

accepted difference: cuml-rf-gpu class_weight: None on every arm is unit class weights, the value each library defines for None

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 9.1 | 9.1..9.1 | 1 | - | - | - | auc=0.617838, auc_matches_fit=True, logloss=0.525953, logloss_matches_fit=True | yes | UNKNOWN | ok |
| cuml-rf-gpu | test | 500000 | 9.0 | 9.0..9.0 | 1 | 1.014 | - | - | auc=0.617836, auc_matches_fit=True, logloss=0.525954, logloss_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 18.8 | 18.8..18.8 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| cuml-rf-gpu | large | 1000000 | 19.2 | 19.2..19.2 | 1 | 0.978 | - | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, cuml-rf-gpu: cuml RandomForest.predict_proba(host X) (FIL conversion refused: 'RandomForestClassifier' object has no attribute 'convert_to_fil_model')

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.dbscan.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 542218.2 | 542218.2..542218.2 | 1 | - | - | - | 5632.2 | 8618.0 | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 66379.1 | 66379.1..66379.1 | 1 | 8.169 | - | - | 2599.0 | 1272.0 | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "rbc" |
| eps | 3.0 | 3.0 |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu algorithm: an exact eps search on every arm: ours 'rbc', cuml-gpu 'brute' (cuml-gpu-rbc races 'rbc')

### dbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.dbscan.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('dbscan: the ball-cover neighbourhood has -1799116104 edges in one batch, which does not fit the int32 CSR this implementation uses. cuML requires int64 labels for RBC (runner.cuh) |
| cuml-gpu | cuml | gpu | opponent | 440431.5 | 440431.5..440431.5 | 1 | - | - | - | 850.3 | 474.0 | n_clusters=36, noise_fraction=0.000174, rows=1000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "rbc" |
| eps | 3.0 | 3.0 |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: cuml-gpu algorithm: an exact eps search on every arm: ours 'rbc', cuml-gpu 'brute' (cuml-gpu-rbc races 'rbc')

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| cuml-gpu | cuml | gpu | opponent | 862.1 | 862.1..862.1 | 1 | - | - | - | 1864.3 | 526.0 | n_clusters=53, noise_fraction=0.256280, rows=100000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| allow_single_cluster | false | false |
| alpha | 1.0 | 1.0 |
| cluster_selection_epsilon | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" |
| max_cluster_size | 0 | 0 |
| metric | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 |
| min_samples | 10 | 10 |
| p | null | - |
| seed | "none (deterministic)" | "none (deterministic)" |

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| cuml-gpu | cuml | gpu | opponent | 371.2 | 371.2..371.2 | 1 | - | - | - | 969.5 | 448.0 | n_clusters=159, noise_fraction=0.130970, rows=100000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| allow_single_cluster | false | false |
| alpha | 1.0 | 1.0 |
| cluster_selection_epsilon | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" |
| max_cluster_size | 0 | 0 |
| metric | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 |
| min_samples | 10 | 10 |
| p | null | - |
| seed | "none (deterministic)" | "none (deterministic)" |

### kde / istella (rows full, shape 100000x220)

race: done, driver rc 0, log `logs/classical.kde.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 43.7 | 43.7..43.7 | 1 | - | - | - | 1580.8 | 1450.0 | mean_log_likelihood=-222.270586, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 6.9 | 6.9..6.9 | 1 | 6.345 | - | - | 1001.5 | 514.0 | mean_log_likelihood=-222.270582, rows_without_density=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | - | "auto" |
| atol | - | 0.0 |
| bandwidth | 1.0 | 1.0 |
| breadth_first | - | true |
| kernel | "gaussian" | "gaussian" |
| leaf_size | - | 40 |
| metric | "euclidean" | "euclidean" |
| rtol | - | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" |

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc 0, log `logs/classical.kde.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 34.3 | 34.3..34.3 | 1 | - | - | - | 1499.6 | 1450.0 | mean_log_likelihood=-14.826460, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 1.9 | 1.9..1.9 | 1 | 18.194 | - | - | 800.2 | 436.0 | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | - | "auto" |
| atol | - | 0.0 |
| bandwidth | 1.0 | 1.0 |
| breadth_first | - | true |
| kernel | "gaussian" | "gaussian" |
| leaf_size | - | 40 |
| metric | "euclidean" | "euclidean" |
| rtol | - | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" |

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 800.9 | 800.9..800.9 | 1 | - | - | - | 3647.9 | 2474.0 | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 483.4 | 483.4..483.4 | 1 | 1.657 | - | - | 4957.5 | 2154.0 | inertia=6.111e+17, inertia_over_ours=1.009914, n_iter=20 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 857.6 | 857.6..857.6 | 1 | 0.934 | - | - | 3168.9 | 3468.9 | inertia=5.991e+17, inertia_over_ours=0.990156, n_iter=55 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | torch (declared) |
| init | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 300 | 300 | 300 |
| metric | - | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 | - |
| seed | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 80.0 | 80.0..80.0 | 1 | - | - | - | eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | 500000 | 6.3 | 6.3..6.3 | 1 | 12.630 | - | - | agreement_vs_ours=0.766430, bits_equal_vs_ours=False, eval_inertia=1.442e+17, label_agreement_own_centers=0.999998 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu | Xq | 500000 | 1.3 | 1.3..1.3 | 1 | 62.616 | - | - | agreement_vs_ours=0.038610, bits_equal_vs_ours=False, eval_inertia=1.402e+17, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml KMeans.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 323.9 | 323.9..323.9 | 1 | - | - | - | 1725.4 | 938.0 | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 390.3 | 390.3..390.3 | 1 | 0.830 | - | - | 1227.3 | 614.0 | inertia=3.093e+08, inertia_over_ours=1.000059, n_iter=42 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 891.5 | 891.5..891.5 | 1 | 0.363 | - | - | 1254.7 | 438.0 | inertia=3.129e+08, inertia_over_ours=1.011537, n_iter=85 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | torch (declared) |
| init | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 300 | 300 | 300 |
| metric | - | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 | - |
| seed | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 11.0 | 11.0..11.0 | 1 | - | - | - | eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | 500000 | 2.8 | 2.8..2.8 | 1 | 3.895 | - | - | agreement_vs_ours=4e-06, bits_equal_vs_ours=False, eval_inertia=4.821e+07, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | 20.109 | - | - | agreement_vs_ours=0.540476, bits_equal_vs_ours=False, eval_inertia=4.922e+07, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml KMeans.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### knn / istella (rows full, shape 400000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 44.0 | 44.0..44.0 | 1 | - | - | - | 1841.0 | 2218.0 | recall_at_k=0.976250, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 73.2 | 73.2..73.2 | 1 | 0.601 | - | - | 1623.0 | 772.0 | recall_at_k=0.976402, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 115.5 | 115.5..115.5 | 1 | 0.381 | - | - | 1219.7 | 3816.8 | recall_at_k=0.981012, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 | 64 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24.5 | 24.5..24.5 | 1 | - | - | - | 1519.2 | 1706.0 | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 40.3 | 40.3..40.3 | 1 | 0.608 | - | - | 818.7 | 452.0 | recall_at_k=0.999742, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 96.4 | 96.4..96.4 | 1 | 0.254 | - | - | 895.3 | 3174.8 | recall_at_k=0.999773, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | torch-gpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 | 64 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.ols.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 910.8 | 910.8..910.8 | 1 | - | - | - | 5261.5 | 5802.0 | finite=True, r2=0.331944, rmse=0.682027 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 79.4 | 79.4..79.4 | 1 | 11.467 | - | - | 5076.3 | 2154.0 | finite=True, r2=-11031.855105, rmse=87.647429 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 582.0 | 582.0..582.0 | 1 | 1.565 | - | - | 3013.1 | 8894.4 | finite=False, r2=nan, rmse=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 19.9 | 19.9..19.9 | 1 | 45.688 | - | - | 3044.6 | 3455.2 | finite=True, r2=0.151604, rmse=0.768590 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | torch-gpu | torch-gpu-eigh |
|---|---||---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | torch (declared) | torch (declared) |
| algorithm | "eig" | - | - | - |
| fit_intercept | true | true | true | true |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 29.5 | 29.5..29.5 | 1 | - | - | - | predict_max_rel_err_own_fp64=9.581e-07, r2_eval=0.331944, rmse_eval=0.682027 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | 500000 | 3.9 | 3.9..3.9 | 1 | 7.478 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=5603.259324, predict_max_rel_err_own_fp64=0.001281, r2_eval=-11032.159107, rmse_eval=87.648636 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu | Xq | 500000 | 1.0 | 1.0..1.0 | 1 | 31.003 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=nan, predict_max_rel_err_own_fp64=nan, r2_eval=nan, rmse_eval=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu-eigh | Xq | 500000 | 1.0 | 1.0..1.0 | 1 | 30.509 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=6.699504, predict_max_rel_err_own_fp64=7.288e-07, r2_eval=0.151604, rmse_eval=0.768590 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml LinearRegression.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 260.5 | 260.5..260.5 | 1 | - | - | - | 1797.8 | 1194.0 | finite=True, r2=0.908837, rmse=4.696466 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 22.7 | 22.7..22.7 | 1 | 11.481 | - | - | 1380.9 | 614.0 | finite=True, r2=0.908836, rmse=4.696488 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 39.1 | 39.1..39.1 | 1 | 6.660 | - | - | 1081.1 | 4649.6 | finite=True, r2=0.908836, rmse=4.696480 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | 132.241 | - | - | 1098.0 | 376.4 | finite=True, r2=0.908836, rmse=4.696490 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | torch-gpu | torch-gpu-eigh |
|---|---||---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | torch (declared) | torch (declared) |
| algorithm | "eig" | - | - | - |
| fit_intercept | true | true | true | true |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 2.7 | 2.7..2.7 | 1 | - | - | - | predict_max_rel_err_own_fp64=8.387e-08, r2_eval=0.908837, rmse_eval=4.696466 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | 500000 | 1.3 | 1.3..1.3 | 1 | 2.133 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.066647, predict_max_rel_err_own_fp64=8.179e-08, r2_eval=0.908836, rmse_eval=4.696488 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu | Xq | 500000 | 0.4 | 0.4..0.4 | 1 | 6.863 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.058064, predict_max_rel_err_own_fp64=1.093e-07, r2_eval=0.908836, rmse_eval=4.696479 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu-eigh | Xq | 500000 | 0.4 | 0.4..0.4 | 1 | 6.767 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.074196, predict_max_rel_err_own_fp64=1.444e-07, r2_eval=0.908836, rmse_eval=4.696490 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml LinearRegression.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 231.4 | 231.4..231.4 | 1 | - | - | - | 3641.2 | 5802.0 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 16.9 | 16.9..16.9 | 1 | 13.728 | - | - | 2997.8 | 3439.6 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 78.7 | 78.7..78.7 | 1 | 2.940 | - | - | 5058.6 | 2190.0 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (stored (measured 2026-09-29T19:19:28Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| n_components | 10 | 10 |
| seed | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | - |
| whiten | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 45.5 | 45.5..45.5 | 1 | - | - | - | transform_max_rel_err_own_fp64=3.75e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 2.3 | 2.3..2.3 | 1 | 19.538 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=3.645e+06, transform_max_rel_err_own_fp64=4.129e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| cuml-gpu | Xq | 500000 | 10.4 | 10.4..10.4 | 1 | 3.574 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=112661.685547, transform_max_rel_err_own_fp64=3.694e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

inference call, cuml-gpu: cuml PCA.transform(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12.7 | 12.7..12.7 | 1 | - | - | - | 1700.9 | 938.0 | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-gpu | torch | gpu | opponent | 1.4 | 1.4..1.4 | 1 | 8.818 | - | - | 1058.1 | 344.4 | explained_variance_ratio_sum=0.999996 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 19.3 | 19.3..19.3 | 1 | 0.656 | - | - | 1309.2 | 642.0 | explained_variance_ratio_sum=0.999996 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (stored (measured 2026-09-29T19:19:00Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-gpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | torch (declared) |
| n_components | 10 | 10 |
| seed | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | - |
| whiten | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 22.8 | 22.8..22.8 | 1 | - | - | - | transform_max_rel_err_own_fp64=1.085e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.6 | 0.6..0.6 | 1 | 39.243 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=29.157851, transform_max_rel_err_own_fp64=1.072e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| cuml-gpu | Xq | 500000 | 1.8 | 1.8..1.8 | 1 | 8.171 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.004157, transform_max_rel_err_own_fp64=1.067e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

inference call, cuml-gpu: cuml PCA.transform(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

### svc / istella (rows full, shape 10000x220)

race: done, driver rc 0, log `logs/classical.svc.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 36.5 | 36.5..36.5 | 1 | - | - | - | 1540.2 | 908.0 | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 23.0 | 23.0..23.0 | 1 | 1.589 | - | - | 970.1 | 464.0 | accuracy=0.922200, n_support=2401 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| seed | 7 | null |
| tol | 0.001 | 0.001 |

accepted difference: ours seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 10000 | 18.1 | 18.1..18.1 | 1 | - | - | - | accuracy_eval=0.922200 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | 10000 | 13.1 | 13.1..13.1 | 1 | 1.382 | - | - | accuracy_eval=0.922200, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml SVC.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc 0, log `logs/classical.svc.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 649.1 | 649.1..649.1 | 1 | - | - | - | 1501.3 | 908.0 | accuracy=0.767500, n_support=5527 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 377.9 | 377.9..377.9 | 1 | 1.718 | - | - | 1028.3 | 440.0 | accuracy=0.767500, n_support=5541 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| seed | 7 | null |
| tol | 0.001 | 0.001 |

accepted difference: ours seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 10000 | 5.2 | 5.2..5.2 | 1 | - | - | - | accuracy_eval=0.767500 | yes | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | 10000 | 5.3 | 5.3..5.3 | 1 | 0.985 | - | - | accuracy_eval=0.767500, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, cuml-gpu: cuml SVC.predict(Xq on the device, output_type cupy); Xq uploaded before the clock, which ends at the device synchronize

## Classical, wave 2

### agglomerative / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.agglomerative.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 96.5 | 96.5..96.5 | 1 | - | - | - | 2430.7 | 938.0 | n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 56.5 | 56.5..56.5 | 1 | 1.709 | - | - | 1695.4 | 440.0 | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.716728 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

config: cuML benchmark (RAPIDS), AgglomerativeClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (attributes) |
| linkage | "single" | "single" |
| metric | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| seed | "none (deterministic)" | "none (deterministic)" |

### agglomerative / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.agglomerative.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21.4 | 21.4..21.4 | 1 | - | - | - | 1545.6 | 938.0 | n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 71.4 | 71.4..71.4 | 1 | 0.299 | - | - | 804.4 | 430.0 | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.685524 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

config: cuML benchmark (RAPIDS), AgglomerativeClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (attributes) |
| linkage | "single" | "single" |
| metric | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| seed | "none (deterministic)" | "none (deterministic)" |

### arima / synthetic (rows full, shape Yfit 64x2000; Yhold 64x100)

race: done, driver rc 0, log `logs/classical2.arima.synthetic.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 547.1 | 547.1..547.1 | 1 | - | - | - | 1497.1 | 762.0 | forecast_rmse=1.515518, insample_rmse=0.999342, mean_aic=5680.976967, mean_llf=-2836.488483 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 785.5 | 785.5..785.5 | 1 | 0.696 | - | - | 841.1 | 812.0 | forecast_rmse=1.515427, insample_rmse=0.999338, mean_aic=5680.957154, mean_llf=-2836.478577 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 143.3 | 143.3..143.3 | 1 | 3.818 | - | - | 196.5 | - | forecast_rmse=1.515423, insample_rmse=0.999338, mean_aic=5680.957160, mean_llf=-2836.478580 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: order=(1,0,1), seasonal_order=(0,0,0,0), trend='c' (cuML fit_intercept=True), maxiter=1000, maximum likelihood. Rows: 64 synthetic ARMA(1,1) series, 2000 fit points, 100 held out. Timed: fit of every series.

mismatch: ours and cuML fit the whole batch in one call; statsmodels fits one series per call (the state-space model, L-BFGS), spread over every core with joblib

mismatch: statsmodels enforce_stationarity and enforce_invertibility at its default (True); ours and cuML have no such parameter

mismatch: seed: no arm has a seed argument (maximum likelihood)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | ? (unreadable (NotImplementedError('ARIMA is unable to be cloned via `get_params` and `set_params`.'))) | mojolearn (get_params) | statsmodels (declared) |
| max_iter | - | 1000 | 1000 |
| order | - | [1, 0, 1] | [1, 0, 1] |
| seasonal_order | - | [0, 0, 0, 0] | [0, 0, 0, 0] |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | - | "c" | "c" |

### elasticnet / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2316.4 | 2316.4..2316.4 | 1 | - | - | - | 3252.0 | 1450.0 | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 316.2 | 316.2..316.2 | 1 | 7.327 | - | - | 2909.6 | 1374.0 | finite=True, r2=0.260922, rmse=0.718134 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 0.1 | 0.1 |
| fit_intercept | true | true |
| l1_ratio | 0.5 | 0.5 |
| max_iter | 1000 | 1000 |
| positive | - | false |
| precompute | - | false |
| seed | "none (deterministic)" | null |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | "cd" |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

### elasticnet / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 58.8 | 58.8..58.8 | 1 | - | - | - | 1576.2 | 682.0 | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 5.9 | 5.9..5.9 | 1 | 10.017 | - | - | 959.3 | 498.0 | finite=True, r2=0.907378, rmse=4.847224 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 0.1 | 0.1 |
| fit_intercept | true | true |
| l1_ratio | 0.5 | 0.5 |
| max_iter | 1000 | 1000 |
| positive | - | false |
| precompute | - | false |
| seed | "none (deterministic)" | null |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | "cd" |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

### ets / synthetic (rows full, shape Yfit 64x1440; Yhold 64x48)

race: done, driver rc 0, log `logs/classical2.ets.synthetic.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 157.2 | 157.2..157.2 | 1 | - | - | - | 1501.4 | 682.0 | forecast_rmse=0.984392, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 489.5 | 489.5..489.5 | 1 | 0.321 | - | - | 924.3 | 480.0 | forecast_rmse=1.073814 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| statsmodels-cpu | statsmodels | cpu | opponent | 459.8 | 459.8..459.8 | 1 | 0.342 | - | - | 195.7 | - | forecast_rmse=0.984664, insample_rmse=0.992056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: trend additive, seasonal additive, seasonal_periods=24, initialization_method='estimated'; ours and cuML start_periods=2, eps=2.24e-3; statsmodels damped_trend=False, use_boxcox=False. Rows: 64 synthetic hourly series, period 24, 1440 fit points, 48 held out. Timed: construct + fit of every series.

mismatch: initialization: ours 'estimated' (its default, statsmodels' definition), statsmodels 'estimated'; cuML has only its heuristic start (start_periods=2), so its row fits the older initialization

mismatch: cuML returns no in-sample predictions; that quality cell is empty

mismatch: trend: ours and cuML are additive-trend with no parameter; statsmodels trend='additive'. eps is ours' and cuML's only; statsmodels fit() uses its own optimizer

mismatch: seed: no arm has a seed argument

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (attributes) | statsmodels (declared) |
| damped_trend | - | - | false |
| eps | 0.00224 | 0.00224 | - |
| initialization_method | - | "estimated" | "estimated" |
| seasonal | "additive" | "additive" | "additive" |
| seasonal_periods | 24 | 24 | 24 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| start_periods | 2 | 2 | - |
| trend | - | - | "additive" |

### gmm / istella (rows full, shape X 100000x200; Xq 20000x200; _dropped_constant_columns 20)

race: done, driver rc 0, log `logs/classical2.gmm.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 974.2 | 974.2..974.2 | 1 | - | - | - | 2597.6 | 1164.0 | bic=-3.851e+07, mean_log_likelihood=200.794403, n_iter=24 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 48820.9 | 48820.9..48820.9 | 1 | 0.020 | - | - | 1468.1 | - | bic=-3.901e+07, mean_log_likelihood=200.768138, n_iter=39 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| covariance_type | "full" | "full" |
| init_params | "kmeans" | "kmeans" |
| max_iter | 100 | 100 |
| n_components | 8 | 8 |
| n_init | 1 | 1 |
| reg_covar | 0.003 | 0.003 |
| seed | 7 | 7 |
| tol | 0.001 | 0.001 |

### gmm / taxi (rows full, shape X 100000x11; Xq 20000x11)

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 180.7 | 180.7..180.7 | 1 | - | - | - | 1567.8 | 682.0 | bic=-3.67e+06, mean_log_likelihood=12.861940, n_iter=32 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host not sampled; GPU not sampled

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| covariance_type | "full" | "full" |
| init_params | "kmeans" | "kmeans" |
| max_iter | 100 | 100 |
| n_components | 8 | 8 |
| n_init | 1 | 1 |
| reg_covar | 1e-06 | 1e-06 |
| seed | 7 | 7 |
| tol | 0.001 | 0.001 |

### gpc / istella (rows full, shape X 3000x220; Xq 3000x220; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpc.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3108.1 | 3108.1..3108.1 | 1 | - | - | - | 2586.3 | 908.0 | accuracy=0.901333, logloss=0.232590, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6759.0 | 6759.0..6759.0 | 1 | 0.460 | - | - | 1619.0 | - | accuracy=0.901333, logloss=0.232597, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)), optimizer=None, max_iter_predict=100, n_restarts_optimizer=0. Rows: 3000 fit and 3000 held-out stride rows of the cls block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: seed: ours refuses random_state (optimizer=None draws nothing); scikit-learn random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| kernel | "(ConstantKernel(1.0) * RBF([14.832396974191326]))" | "1**2 * RBF(length_scale=14.8)" |
| max_iter_predict | 100 | 100 |
| n_restarts_optimizer | 0 | 0 |
| seed | null | 7 |

accepted difference: ours seed: mojolearn GaussianProcessClassifier refuses random_state (optimizer=None draws nothing); scikit-learn gets 7

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### gpc / taxi (rows full, shape X 3000x11; Xq 3000x11; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpc.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2376.7 | 2376.7..2376.7 | 1 | - | - | - | 1700.2 | 908.0 | accuracy=0.761000, logloss=0.541286, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2503.3 | 2503.3..2503.3 | 1 | 0.949 | - | - | 734.4 | - | accuracy=0.761000, logloss=0.541358, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)), optimizer=None, max_iter_predict=100, n_restarts_optimizer=0. Rows: 3000 fit and 3000 held-out stride rows of the cls block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: seed: ours refuses random_state (optimizer=None draws nothing); scikit-learn random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| kernel | "(ConstantKernel(1.0) * RBF([3.3166247903554]))" | "1**2 * RBF(length_scale=3.32)" |
| max_iter_predict | 100 | 100 |
| n_restarts_optimizer | 0 | 0 |
| seed | null | 7 |

accepted difference: ours seed: mojolearn GaussianProcessClassifier refuses random_state (optimizer=None draws nothing); scikit-learn gets 7

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### gpr / istella (rows full, shape X 3000x220; Xq 3000x220; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpr.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 583.1 | 583.1..583.1 | 1 | - | - | - | 2623.9 | 908.0 | finite=True, mean_log_predictive_density=-9.285754, r2=0.235346, rmse=0.760439 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1486.5 | 1486.5..1486.5 | 1 | 0.392 | - | - | 1412.4 | - | finite=True, mean_log_predictive_density=-9.287148, r2=0.235368, rmse=0.760428 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)) + WhiteKernel(noise_level=0.01), alpha=2**-20 (the one ridge IDENTICAL accepts beside 0; the noise lives in the WhiteKernel so the float32 factor of K exists on taxi's near-duplicate rows), optimizer=None, normalize_y=False, n_restarts_optimizer=0, random_state=7. Rows: 3000 fit and 3000 held-out stride rows of the reg block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: kernel: the same kernel built from each library's own classes

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 9.5367431640625e-07 | 9.5367431640625e-07 |
| kernel | "((ConstantKernel(1.0) * RBF([14.832396974191326])) + WhiteKernel(0.01))" | "1**2 * RBF(length_scale=14.8) + WhiteKernel(noise_level=0.01)" |
| n_restarts_optimizer | 0 | 0 |
| seed | 7 | 7 |

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### gpr / taxi (rows full, shape X 3000x11; Xq 3000x11; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpr.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 498.8 | 498.8..498.8 | 1 | - | - | - | 1781.9 | 908.0 | finite=True, mean_log_predictive_density=-311.458394, r2=0.889630, rmse=5.041639 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 304.8 | 304.8..304.8 | 1 | 1.636 | - | - | 523.7 | - | finite=True, mean_log_predictive_density=-311.539594, r2=0.889629, rmse=5.041653 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)) + WhiteKernel(noise_level=0.01), alpha=2**-20 (the one ridge IDENTICAL accepts beside 0; the noise lives in the WhiteKernel so the float32 factor of K exists on taxi's near-duplicate rows), optimizer=None, normalize_y=False, n_restarts_optimizer=0, random_state=7. Rows: 3000 fit and 3000 held-out stride rows of the reg block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: kernel: the same kernel built from each library's own classes

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 9.5367431640625e-07 | 9.5367431640625e-07 |
| kernel | "((ConstantKernel(1.0) * RBF([3.3166247903554])) + WhiteKernel(0.01))" | "1**2 * RBF(length_scale=3.32) + WhiteKernel(noise_level=0.01)" |
| n_restarts_optimizer | 0 | 0 |
| seed | 7 | 7 |

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/classical2.ivf.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 525639.9 | 525639.9..525639.9 | 1 | - | - | - | 3579.2 | 1450.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuvs-gpu | cuvs | gpu | opponent | 429.5 | 429.5..429.5 | 1 | 1223.980 | - | - | 1726.9 | 770.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 |
| nlist | 1024 | 1024 |
| nprobe | 32 | 32 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 |

accepted difference: cuvs-gpu seed: cuVS ivf_flat IndexParams takes no seed and its k-means training samples rows; ours and faiss get 7

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/classical2.ivf.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 34850.7 | 34850.7..34850.7 | 1 | - | - | - | 1610.6 | 682.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuvs-gpu | cuvs | gpu | opponent | 219.5 | 219.5..219.5 | 1 | 158.802 | - | - | 931.0 | 450.0 | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuvs-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuvs-gpu | ours |
|---|---||---|---|
| library (source) | cuvs (declared) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 |
| nlist | 1024 | 1024 |
| nprobe | 32 | 32 |
| seed | "none (no argument; draws random numbers, see exceptions)" | 7 |

accepted difference: cuvs-gpu seed: cuVS ivf_flat IndexParams takes no seed and its k-means training samples rows; ours and faiss get 7

### kernel-ridge / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4913.7 | 4913.7..4913.7 | 1 | - | - | - | 2456.6 | 1676.0 | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 326.0 | 326.0..326.0 | 1 | 15.071 | - | - | 1804.1 | 494.0 | finite=True, r2=0.385427, rmse=0.646407 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| seed | "none (deterministic)" | "none (deterministic)" |

### kernel-ridge / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4910.5 | 4910.5..4910.5 | 1 | - | - | - | 1547.7 | 1420.0 | finite=True, r2=0.726543, rmse=8.330373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 326.2 | 326.2..326.2 | 1 | 15.055 | - | - | 1045.9 | 474.0 | finite=True, r2=0.726543, rmse=8.330375 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| seed | "none (deterministic)" | "none (deterministic)" |

### knn-clf / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 51.5 | 51.5..51.5 | 1 | - | - | - | 2599.9 | 938.0 | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 69.9 | 69.9..69.9 | 1 | 0.737 | - | - | 2056.3 | 602.0 | accuracy=0.926250 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### knn-clf / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 45.0 | 45.0..45.0 | 1 | - | - | - | 1560.1 | 682.0 | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 8.6 | 8.6..8.6 | 1 | 5.212 | - | - | 978.8 | 548.0 | accuracy=0.741750 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### knn-reg / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 32.4 | 32.4..32.4 | 1 | - | - | - | 2594.9 | 938.0 | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 38.5 | 38.5..38.5 | 1 | 0.841 | - | - | 1958.2 | 602.0 | finite=True, r2=0.418145, rmse=0.625388 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### knn-reg / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11.5 | 11.5..11.5 | 1 | - | - | - | 1554.8 | 682.0 | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 5.2 | 5.2..5.2 | 1 | 2.230 | - | - | 837.1 | 440.0 | finite=True, r2=0.937323, rmse=3.842028 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "brute" | "brute" |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### lasso / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5731.4 | 5731.4..5731.4 | 1 | - | - | - | 3250.9 | 1450.0 | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 953.2 | 953.2..953.2 | 1 | 6.013 | - | - | 2909.8 | 1374.0 | finite=True, r2=0.310837, rmse=0.693460 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 0.01 | 0.01 |
| fit_intercept | true | true |
| max_iter | 1000 | 1000 |
| positive | - | false |
| precompute | - | false |
| seed | "none (deterministic)" | null |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | "cd" |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

### lasso / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 63.5 | 63.5..63.5 | 1 | - | - | - | 1581.0 | 682.0 | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 7.1 | 7.1..7.1 | 1 | 8.972 | - | - | 959.1 | 498.0 | finite=True, r2=0.908995, rmse=4.804745 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 0.01 | 0.01 |
| fit_intercept | true | true |
| max_iter | 1000 | 1000 |
| positive | - | false |
| precompute | - | false |
| seed | "none (deterministic)" | null |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | "cd" |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

### linearsvc / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1198.3 | 1198.3..1198.3 | 1 | - | - | - | 2487.5 | 1450.0 | accuracy=0.923480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 258.1 | 258.1..258.1 | 1 | 4.642 | - | - | 2995.8 | 1366.0 | accuracy=0.922880 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| loss | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

### linearsvc / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 143.0 | 143.0..143.0 | 1 | - | - | - | 1607.1 | 682.0 | accuracy=0.763330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 8.3 | 8.3..8.3 | 1 | 17.188 | - | - | 1078.8 | 490.0 | accuracy=0.762990 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| loss | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

### linearsvr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1558.3 | 1558.3..1558.3 | 1 | - | - | - | 2423.9 | 1450.0 | finite=True, r2=-0.106754, rmse=0.878792 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 84.0 | 84.0..84.0 | 1 | 18.548 | - | - | 2901.5 | 1374.0 | finite=True, r2=-0.106761, rmse=0.878795 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 |
| fit_intercept | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

### linearsvr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 59.5 | 59.5..59.5 | 1 | - | - | - | 1546.7 | 682.0 | finite=True, r2=0.899813, rmse=5.041302 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 18.9 | 18.9..18.9 | 1 | 3.139 | - | - | 993.1 | 498.0 | finite=True, r2=0.899814, rmse=5.041286 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 |
| fit_intercept | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | false |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.0001 | 0.0001 |

### logreg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6282.2 | 6282.2..6282.2 | 1 | - | - | - | 2426.3 | 1450.0 | accuracy=0.924540, logloss=0.181245, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 2136.2 | 2136.2..2136.2 | 1 | 2.941 | - | - | 3012.6 | 1374.0 | accuracy=0.924550, logloss=0.181262, nonfinite_proba_rows=0 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| l1_ratio | null | null |
| max_iter | 1000 | 1000 |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "qn" | "qn" |
| tol | 0.0001 | 0.0001 |

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: cuml-gpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

### logreg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 91.3 | 91.3..91.3 | 1 | - | - | - | 1548.8 | 682.0 | accuracy=0.763340, logloss=0.538984, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 32.1 | 32.1..32.1 | 1 | 2.842 | - | - | 1078.7 | 498.0 | accuracy=0.763340, logloss=0.538984, nonfinite_proba_rows=0 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| l1_ratio | null | null |
| max_iter | 1000 | 1000 |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "qn" | "qn" |
| tol | 0.0001 | 0.0001 |

accepted difference: cuml-gpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: cuml-gpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

### nystroem / istella (rows full, shape X 100000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.nystroem.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1477.1 | 1477.1..1477.1 | 1 | - | - | - | 1731.3 | - | kernel_rel_error=0.038958 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the landmark rows are each library's own random draw from seed 7

mismatch: degree=3, coef0=1.0 on every arm; the rbf kernel reads neither

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| n_components | 256 | 256 |
| seed | 7 | 7 |

### nystroem / taxi (rows full, shape X 100000x11; Xcheck 1000x11)

race: done, driver rc 0, log `logs/classical2.nystroem.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 912.6 | 912.6..912.6 | 1 | - | - | - | 693.7 | - | kernel_rel_error=0.044369 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the landmark rows are each library's own random draw from seed 7

mismatch: degree=3, coef0=1.0 on every arm; the rbf kernel reads neither

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| n_components | 256 | 256 |
| seed | 7 | 7 |

### rbf-sampler / istella (rows full, shape X 100000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.rbf-sampler.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 174.8 | 174.8..174.8 | 1 | - | - | - | 2793.3 | 908.0 | kernel_rel_error=0.141980 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 99.3 | 99.3..99.3 | 1 | 1.760 | - | - | 1463.4 | - | kernel_rel_error=0.137405 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the random Fourier features are each library's own draw from seed 7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| n_components | 256 | 256 |
| seed | 7 | 7 |

### rbf-sampler / taxi (rows full, shape X 100000x11; Xcheck 1000x11)

race: done, driver rc 0, log `logs/classical2.rbf-sampler.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 83.7 | 83.7..83.7 | 1 | - | - | - | 1741.2 | 908.0 | kernel_rel_error=0.108549 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 38.9 | 38.9..38.9 | 1 | 2.150 | - | - | 440.4 | - | kernel_rel_error=0.083775 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the random Fourier features are each library's own draw from seed 7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| n_components | 256 | 256 |
| seed | 7 | 7 |

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.ridge.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 799.8 | 799.8..799.8 | 1 | - | - | - | 3233.7 | 4010.0 | finite=True, r2=0.328682, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 48.6 | 48.6..48.6 | 1 | 16.467 | - | - | 2961.1 | 1410.0 | finite=True, r2=-0.251259, rmse=0.934403 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 1.0 | 1.0 |
| fit_intercept | true | true |
| max_iter | null | - |
| normalize | - | false |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "eig" | "eig" |
| tol | 0.0001 | - |

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.ridge.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 77.0 | 77.0..77.0 | 1 | - | - | - | 1550.8 | 682.0 | finite=True, r2=0.908983, rmse=4.805042 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 9.4 | 9.4..9.4 | 1 | 8.227 | - | - | 990.5 | 534.0 | finite=True, r2=0.908983, rmse=4.805051 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| alpha | 1.0 | 1.0 |
| fit_intercept | true | true |
| max_iter | null | - |
| normalize | - | false |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "eig" | "eig" |
| tol | 0.0001 | - |

### spectral-embedding / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 133.9 | 133.9..133.9 | 1 | - | - | - | 1535.2 | 682.0 | trustworthiness_k15=0.799386 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 141.8 | 141.8..141.8 | 1 | 0.945 | - | - | 1011.7 | 496.0 | trustworthiness_k15=0.882904 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 18223.3 | 18223.3..18223.3 | 1 | 0.007 | - | - | 468.8 | - | trustworthiness_k15=0.461447 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows: the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| gamma | - | null | null |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral-embedding / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 70.7 | 70.7..70.7 | 1 | - | - | - | 1511.9 | 682.0 | trustworthiness_k15=0.884889 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 73.2 | 73.2..73.2 | 1 | 0.966 | - | - | 995.2 | 478.0 | trustworthiness_k15=0.891595 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 5264.2 | 5264.2..5264.2 | 1 | 0.013 | - | - | 244.2 | - | trustworthiness_k15=0.898012 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows: the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| gamma | - | null | null |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.spectral.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 92.3 | 92.3..92.3 | 1 | - | - | - | 2469.6 | 682.0 | n_clusters=8, silhouette=0.147668 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 123.0 | 123.0..123.0 | 1 | 0.750 | - | - | 1929.8 | 490.0 | ari_vs_ours=0.999106, n_clusters=8, silhouette=0.147757 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2759.0 | 2759.0..2759.0 | 1 | 0.033 | - | - | 1224.8 | - | ari_vs_ours=0.999826, n_clusters=8, silhouette=0.147699 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, random_state=42 (the cuML benchmark's SpectralClustering), assign_labels='kmeans', n_components=8. Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

config: cuML benchmark (RAPIDS), SpectralClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| assign_labels | - | "kmeans" | "kmeans" |
| coef0 | - | - | 1 |
| degree | - | - | 3 |
| gamma | - | null | 1.0 |
| n_clusters | 8 | 8 | 8 |
| n_components | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| n_neighbors | 10 | 10 | 10 |
| seed | 42 | 42 | 42 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.spectral.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 87.3 | 87.3..87.3 | 1 | - | - | - | 1557.1 | 682.0 | n_clusters=8, silhouette=0.039910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 211.5 | 211.5..211.5 | 1 | 0.413 | - | - | 1039.2 | 480.0 | ari_vs_ours=0.574097, n_clusters=8, silhouette=0.087609 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2015.4 | 2015.4..2015.4 | 1 | 0.043 | - | - | 276.4 | - | ari_vs_ours=0.582313, n_clusters=8, silhouette=0.089894 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, random_state=42 (the cuML benchmark's SpectralClustering), assign_labels='kmeans', n_components=8. Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

config: cuML benchmark (RAPIDS), SpectralClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| assign_labels | - | "kmeans" | "kmeans" |
| coef0 | - | - | 1 |
| degree | - | - | 3 |
| gamma | - | null | 1.0 |
| n_clusters | 8 | 8 | 8 |
| n_components | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| n_neighbors | 10 | 10 | 10 |
| seed | 42 | 42 | 42 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### svr / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.svr.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 65.8 | 65.8..65.8 | 1 | - | - | - | 2472.4 | 908.0 | finite=True, r2=0.318258, rmse=0.680816 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 40.2 | 40.2..40.2 | 1 | 1.635 | - | - | 1795.7 | 462.0 | finite=True, r2=0.318232, rmse=0.680829 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| epsilon | 0.1 | 0.1 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

### svr / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.svr.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56.4 | 56.4..56.4 | 1 | - | - | - | 1551.6 | 1164.0 | finite=True, r2=0.767551, rmse=7.680395 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 44.2 | 44.2..44.2 | 1 | 1.277 | - | - | 903.7 | 442.0 | finite=True, r2=0.767551, rmse=7.680405 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| C | 1.0 | 1.0 |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| epsilon | 0.1 | 0.1 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc 0, log `logs/classical2.tsvd.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 96.4 | 96.4..96.4 | 1 | - | - | - | 2334.4 | 2986.0 | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.002554 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 35.3 | 35.3..35.3 | 1 | 2.730 | - | - | 2738.7 | 1314.0 | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001472 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "full" | "covariance_eigh" |
| n_components | 10 | 10 |
| n_iter | 15 | 5 |
| seed | 7 | 7 |
| tol | 1e-07 | 0.0 |

accepted difference: cuml-gpu algorithm: cuML TruncatedSVD has no 'covariance_eigh'; it runs 'full'

accepted difference: cuml-gpu n_iter: cuML TruncatedSVD's n_iter is read only by its 'jacobi' solver (cuml/decomposition/tsvd.pyx: 'Used in Jacobi solver'); the raced 'full' solver, COV_EIG_DQ, a covariance eigendecomposition, iterates nothing. cuML's own scikit-learn interop maps n_iter 5 to its 15

accepted difference: cuml-gpu tol: cuML TruncatedSVD's tol is read only by its 'jacobi' solver (cuml/decomposition/tsvd.pyx: 'Used if algorithm = "jacobi"'); the raced 'full' solver has no tolerance. cuML's own scikit-learn interop maps tol 0.0 to its 1e-7

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc 0, log `logs/classical2.tsvd.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.7 | 6.7..6.7 | 1 | - | - | - | 1535.4 | 682.0 | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 5.9 | 5.9..5.9 | 1 | 1.134 | - | - | 973.4 | 516.0 | explained_variance_ratio_sum=0.999964, relative_reconstruction_error=0.003257 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| algorithm | "full" | "covariance_eigh" |
| n_components | 10 | 10 |
| n_iter | 15 | 5 |
| seed | 7 | 7 |
| tol | 1e-07 | 0.0 |

accepted difference: cuml-gpu algorithm: cuML TruncatedSVD has no 'covariance_eigh'; it runs 'full'

accepted difference: cuml-gpu n_iter: cuML TruncatedSVD's n_iter is read only by its 'jacobi' solver (cuml/decomposition/tsvd.pyx: 'Used in Jacobi solver'); the raced 'full' solver, COV_EIG_DQ, a covariance eigendecomposition, iterates nothing. cuML's own scikit-learn interop maps n_iter 5 to its 15

accepted difference: cuml-gpu tol: cuML TruncatedSVD's tol is read only by its 'jacobi' solver (cuml/decomposition/tsvd.pyx: 'Used if algorithm = "jacobi"'); the raced 'full' solver has no tolerance. cuML's own scikit-learn interop maps tol 0.0 to its 1e-7

### umap / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.umap.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 368.1 | 368.1..368.1 | 1 | - | - | - | 1572.8 | 682.0 | trustworthiness_k15=0.979906 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 515.2 | 515.2..515.2 | 1 | 0.714 | - | - | 1027.6 | 606.0 | trustworthiness_k15=0.979912 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows: 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

config: cuML benchmark (RAPIDS), UMAP-Unsupervised (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| init | "spectral" | "spectral" |
| learning_rate | 1.0 | 1.0 |
| local_connectivity | 1.0 | 1.0 |
| metric | "euclidean" | "euclidean" |
| min_dist | 0.1 | 0.1 |
| n_components | 2 | 2 |
| n_epochs | 500 | 500 |
| n_neighbors | 5 | 5 |
| negative_sample_rate | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 |
| seed | 7 | 7 |
| set_op_mix_ratio | 1.0 | 1.0 |
| spread | 1.0 | 1.0 |

### umap / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.umap.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 141.7 | 141.7..141.7 | 1 | - | - | - | 1510.9 | 682.0 | trustworthiness_k15=0.990480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| cuml-gpu | cuml | gpu | opponent | 255.0 | 255.0..255.0 | 1 | 0.556 | - | - | 980.8 | 588.0 | trustworthiness_k15=0.992305 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows: 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

config: cuML benchmark (RAPIDS), UMAP-Unsupervised (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours |
|---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) |
| init | "spectral" | "spectral" |
| learning_rate | 1.0 | 1.0 |
| local_connectivity | 1.0 | 1.0 |
| metric | "euclidean" | "euclidean" |
| min_dist | 0.1 | 0.1 |
| n_components | 2 | 2 |
| n_epochs | 500 | 500 |
| n_neighbors | 5 | 5 |
| negative_sample_rate | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 |
| seed | 7 | 7 |
| set_op_mix_ratio | 1.0 | 1.0 |
| spread | 1.0 | 1.0 |

## Neural

### gemm-bf16 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-bf16.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 55.2 | 55.2..55.2 | 1 | - | - | - | 495.8 | 1164.0 | max_rel_err_vs_fp64=1.155e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-bf16 | torch | gpu | opponent | 47.0 | 47.0..47.0 | 1 | 1.174 | - | - | 1190.3 | 176.2 | max_abs_diff_vs_ours=1.000977, max_rel_diff_vs_ours=0.002764, max_rel_err_vs_fp64=0.002764 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 46.6 | 46.6..46.6 | 1 | 1.183 | - | - | 1400.4 | 176.2 | max_abs_diff_vs_ours=1.000977, max_rel_diff_vs_ours=0.002764, max_rel_err_vs_fp64=0.002764 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-eager-bf16 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### gemm-int8 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-int8.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2917.1 | 2917.1..2917.1 | 1 | - | - | - | 385.0 | 682.0 | max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-int8 | torch | gpu | opponent | 48.2 | 48.2..48.2 | 1 | 60.558 | - | - | 803.9 | 160.0 | max_abs_diff_vs_ours=0.000000, max_rel_diff_vs_ours=0.000000, max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-int8 | torch | gpu | opponent | 47.8 | 47.8..47.8 | 1 | 61.040 | - | - | 1010.3 | 160.0 | max_abs_diff_vs_ours=0.000000, max_rel_diff_vs_ours=0.000000, max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-int8, torch-compile-int8: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-int8 | torch-eager-int8 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 56.4 | 56.4..56.4 | 1 | - | - | - | 433.0 | 908.0 | max_rel_err_vs_fp64=2.399e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 51.0 | 51.0..51.0 | 1 | 1.106 | - | - | 934.7 | 200.1 | max_abs_diff_vs_ours=0.0005341, max_rel_diff_vs_ours=1.475e-06, max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 49.4 | 49.4..49.4 | 1 | 1.141 | - | - | 936.8 | 200.1 | max_abs_diff_vs_ours=0.100822, max_rel_diff_vs_ours=0.0002785, max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 52.3 | 52.3..52.3 | 1 | 1.078 | - | - | 1135.0 | 200.1 | max_abs_diff_vs_ours=0.0005341, max_rel_diff_vs_ours=1.475e-06, max_rel_err_vs_fp64=1.401e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 50.3 | 50.3..50.3 | 1 | 1.121 | - | - | 1136.8 | 200.1 | max_abs_diff_vs_ours=0.100822, max_rel_diff_vs_ours=0.0002785, max_rel_err_vs_fp64=0.0002784 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 47.0 | 47.0..47.0 | 1 | 1.200 | - | - | 1169.8 | 240.2 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 49.7 | 49.7..49.7 | 1 | 1.135 | - | - | 1404.6 | 240.2 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-forward.bytes.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 147.4 | 147.4..147.4 | 1 | - | - | - | 1812.5 | 4492.0 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 49.8 | 49.8..49.8 | 1 | 2.957 | - | - | 1061.5 | 173.2 | max_abs_diff_vs_ours=1.371e-06, max_rel_diff_vs_ours=1.882e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 46.5 | 46.5..46.5 | 1 | 3.169 | - | - | 1053.9 | 173.2 | max_abs_diff_vs_ours=0.0007233, max_rel_diff_vs_ours=0.0009929, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 48.5 | 48.5..48.5 | 1 | 3.038 | - | - | 1172.1 | 154.7 | max_abs_diff_vs_ours=1.341e-06, max_rel_diff_vs_ours=1.841e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 50.0 | 50.0..50.0 | 1 | 2.945 | - | - | 1166.8 | 154.7 | max_abs_diff_vs_ours=0.0007184, max_rel_diff_vs_ours=0.0009862, mean_nll=9.018732 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 49.4 | 49.4..49.4 | 1 | 2.984 | - | - | 1181.6 | 192.5 | max_abs_diff_vs_ours=0.006870, max_rel_diff_vs_ours=0.009431, mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 51.2 | 51.2..51.2 | 1 | 2.879 | - | - | 1329.1 | 192.5 | max_abs_diff_vs_ours=0.006304, max_rel_diff_vs_ours=0.008653, mean_nll=9.018669 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

### lm-host-train-step / bytes (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-host-train-step.bytes.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 2941.1 | 2941.1..2941.1 | 1 | - | - | - | 2172.0 | - | loss_first_step=9.017858, loss_last_step=8.367768, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 970.3 | 970.3..970.3 | 1 | 3.031 | - | - | 1479.0 | - | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.017857, loss_last_abs_diff_vs_ours=9.537e-07, loss_last_step=8.367767, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 314.3 | 314.3..314.3 | 1 | 9.357 | - | - | 1486.4 | - | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.017857, loss_last_abs_diff_vs_ours=1.907e-06, loss_last_step=8.367766, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 690.4 | 690.4..690.4 | 1 | 4.260 | - | - | 1357.8 | - | loss_first_abs_diff_vs_ours=0.0001106, loss_first_step=9.017747, loss_last_abs_diff_vs_ours=0.0005798, loss_last_step=8.368348, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 371.5 | 371.5..371.5 | 1 | 7.916 | - | - | 1437.5 | - | loss_first_abs_diff_vs_ours=9.06e-05, loss_first_step=9.017767, loss_last_abs_diff_vs_ours=1.431e-05, loss_last_step=8.367754, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### lm-infer / bytes (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-infer.bytes.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 1096.4 | 1096.4..1096.4 | 1 | - | - | - | 466.8 | - | mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 172.5 | 172.5..172.5 | 1 | 6.356 | - | - | 808.6 | - | max_abs_diff_vs_ours=7.227e-07, max_rel_diff_vs_ours=9.921e-07, mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 72.2 | 72.2..72.2 | 1 | 15.186 | - | - | 1162.0 | - | max_abs_diff_vs_ours=6.706e-07, max_rel_diff_vs_ours=9.205e-07, mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 310.2 | 310.2..310.2 | 1 | 3.535 | - | - | 816.7 | - | max_abs_diff_vs_ours=0.006180, max_rel_diff_vs_ours=0.008484, mean_nll=9.017748 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 77.7 | 77.7..77.7 | 1 | 14.105 | - | - | 1121.6 | - | max_abs_diff_vs_ours=0.006585, max_rel_diff_vs_ours=0.009040, mean_nll=9.017761 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-train-step.bytes.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 46.5 | 46.5..46.5 | 1 | - | - | - | 1595.2 | 3724.0 | loss_first_step=9.018733, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 22.1 | 22.1..22.1 | 1 | 2.106 | - | - | 1261.1 | 959.2 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=9.018733, loss_last_abs_diff_vs_ours=3.815e-06, loss_last_step=8.422415, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 17.7 | 17.7..17.7 | 1 | 2.630 | - | - | 1251.7 | 959.2 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018732, loss_last_abs_diff_vs_ours=9.537e-05, loss_last_step=8.422506, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 19.5 | 19.5..19.5 | 1 | 2.381 | - | - | 1225.6 | 720.7 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018734, loss_last_abs_diff_vs_ours=3.815e-06, loss_last_step=8.422415, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 20.1 | 20.1..20.1 | 1 | 2.309 | - | - | 1222.7 | 727.2 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018732, loss_last_abs_diff_vs_ours=0.0001049, loss_last_step=8.422516, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 18.6 | 18.6..18.6 | 1 | 2.498 | - | - | 1387.8 | 796.6 | loss_first_abs_diff_vs_ours=0.0003309, loss_first_step=9.018402, loss_last_abs_diff_vs_ours=0.0008411, loss_last_step=8.421570, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 11.0 | 11.0..11.0 | 1 | 4.238 | - | - | 1396.9 | 546.6 | loss_first_abs_diff_vs_ours=6.39e-05, loss_first_step=9.018669, loss_last_abs_diff_vs_ours=0.002161, loss_last_step=8.420250, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-forward.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9.9 | 9.9..9.9 | 1 | - | - | - | 196.6 | 908.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 149.6 | 149.6..149.6 | 1 | 0.066 | - | - | 1062.5 | 354.3 | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.945e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 153.5 | 153.5..153.5 | 1 | 0.064 | - | - | 1041.7 | 354.3 | max_abs_diff_vs_ours=3.815e-06, max_rel_diff_vs_ours=1.902e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 195.9 | 195.9..195.9 | 1 | 0.051 | - | - | 1189.8 | 307.1 | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.568e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 |

### mamba1-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-infer.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 80.9 | 80.9..80.9 | 1 | - | - | - | 114.6 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 121.8 | 121.8..121.8 | 1 | 0.665 | - | - | 720.9 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.948e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 240.8 | 240.8..240.8 | 1 | 0.336 | - | - | 812.8 | - | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.57e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-forward.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12.4 | 12.4..12.4 | 1 | - | - | - | 199.3 | 908.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 25.8 | 25.8..25.8 | 1 | 0.479 | - | - | 1192.2 | 3228.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 25.7 | 25.7..25.7 | 1 | 0.481 | - | - | 1181.2 | 3228.4 | max_abs_diff_vs_ours=0.000509, max_rel_diff_vs_ours=0.0001731 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 26.1 | 26.1..26.1 | 1 | 0.474 | - | - | 1250.8 | 3228.4 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 26.5 | 26.5..26.5 | 1 | 0.468 | - | - | 1208.0 | 3228.4 | max_abs_diff_vs_ours=0.000509, max_rel_diff_vs_ours=0.0001731 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 21.7 | 21.7..21.7 | 1 | 0.571 | - | - | 1326.0 | 1692.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 21.7 | 21.7..21.7 | 1 | 0.569 | - | - | 1383.4 | 1692.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

### mamba2-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-infer.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 82.9 | 82.9..82.9 | 1 | - | - | - | 126.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 307.0 | 307.0..307.0 | 1 | 0.270 | - | - | 1428.2 | - | max_abs_diff_vs_ours=7.749e-07, max_rel_diff_vs_ours=2.751e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 291.9 | 291.9..291.9 | 1 | 0.284 | - | - | 1626.6 | - | max_abs_diff_vs_ours=7.749e-07, max_rel_diff_vs_ours=2.751e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 218.8 | 218.8..218.8 | 1 | 0.379 | - | - | 1066.8 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 268.2 | 268.2..268.2 | 1 | 0.309 | - | - | 1234.5 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-forward.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.9 | 7.9..7.9 | 1 | - | - | - | 1504.7 | 908.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 83.7 | 83.7..83.7 | 1 | 0.094 | - | - | 1190.9 | 219.7 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 83.9 | 83.9..83.9 | 1 | 0.094 | - | - | 1186.6 | 219.7 | max_abs_diff_vs_ours=0.0001827, max_rel_diff_vs_ours=7.993e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 13.5 | 13.5..13.5 | 1 | 0.582 | - | - | 3465.6 | 83.6 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 9.3 | 9.3..9.3 | 1 | 0.844 | - | - | 3068.5 | 83.6 | max_abs_diff_vs_ours=0.0001798, max_rel_diff_vs_ours=7.867e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 83.8 | 83.8..83.8 | 1 | 0.094 | - | - | 1313.7 | 212.6 | max_abs_diff_vs_ours=0.002095, max_rel_diff_vs_ours=0.0009164 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 9.1 | 9.1..9.1 | 1 | 0.869 | - | - | 3809.6 | 90.5 | max_abs_diff_vs_ours=0.001982, max_rel_diff_vs_ours=0.0008672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

### mamba3-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-infer.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 73.9 | 73.9..73.9 | 1 | - | - | - | 134.2 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 127.4 | 127.4..127.4 | 1 | 0.580 | - | - | 766.1 | - | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.06e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 66.5 | 66.5..66.5 | 1 | 1.110 | - | - | 1151.0 | - | max_abs_diff_vs_ours=2.98e-07, max_rel_diff_vs_ours=1.324e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 30.5 | 30.5..30.5 | 1 | 2.418 | - | - | 764.1 | - | max_abs_diff_vs_ours=0.002148, max_rel_diff_vs_ours=0.0009547 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 66.3 | 66.3..66.3 | 1 | 1.114 | - | - | 1151.6 | - | max_abs_diff_vs_ours=0.002040, max_rel_diff_vs_ours=0.0009065 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-infer / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-infer.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 0.2 | 0.2..0.2 | 1 | - | - | - | 81.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 29.6 | 29.6..29.6 | 1 | 0.008 | - | - | 515.2 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=1.1e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 15.3 | 15.3..15.3 | 1 | 0.016 | - | - | 834.2 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=1.1e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 16.3 | 16.3..16.3 | 1 | 0.015 | - | - | 534.4 | - | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 12.9 | 12.9..12.9 | 1 | 0.019 | - | - | 848.5 | - | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-train-step.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.4 | 2.4..2.4 | 1 | - | - | - | 1493.4 | 682.0 | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1.8 | 1.8..1.8 | 1 | 1.383 | - | - | 1013.9 | 16.3 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 1.7 | 1.7..1.7 | 1 | 1.474 | - | - | 1016.8 | 16.3 | loss_first_abs_diff_vs_ours=8.941e-06, loss_first_step=1.160392, loss_last_abs_diff_vs_ours=5.484e-06, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2.3 | 2.3..2.3 | 1 | 1.057 | - | - | 1104.6 | 16.3 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 2.4 | 2.4..2.4 | 1 | 1.007 | - | - | 1048.9 | 16.3 | loss_first_abs_diff_vs_ours=8.941e-06, loss_first_step=1.160392, loss_last_abs_diff_vs_ours=5.603e-06, loss_last_step=1.123355, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2.0 | 2.0..2.0 | 1 | 1.214 | - | - | 1228.8 | 16.3 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2.2 | 2.2..2.2 | 1 | 1.122 | - | - | 1295.5 | 16.3 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001007, loss_last_step=1.123462, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-forward.bytes.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12.0 | 12.0..12.0 | 1 | - | - | - | 1596.8 | 900.0 | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 46.1 | 46.1..46.1 | 1 | 0.261 | - | - | 1260.1 | 137.4 | max_abs_diff_vs_ours=2.235e-06, max_rel_diff_vs_ours=1.257e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 46.4 | 46.4..46.4 | 1 | 0.259 | - | - | 1238.9 | 137.4 | max_abs_diff_vs_ours=0.001577, max_rel_diff_vs_ours=0.0008869, mean_nll=5.635948 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 6.0 | 6.0..6.0 | 1 | 2.014 | - | - | 1861.2 | 64.2 | max_abs_diff_vs_ours=2.056e-06, max_rel_diff_vs_ours=1.156e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 5.0 | 5.0..5.0 | 1 | 2.392 | - | - | 1723.9 | 64.2 | max_abs_diff_vs_ours=0.001510, max_rel_diff_vs_ours=0.0008487, mean_nll=5.635956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 48.2 | 48.2..48.2 | 1 | 0.249 | - | - | 1365.6 | 142.8 | max_abs_diff_vs_ours=0.019621, max_rel_diff_vs_ours=0.011031, mean_nll=5.635952 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 5.3 | 5.3..5.3 | 1 | 2.256 | - | - | 2105.1 | 73.6 | max_abs_diff_vs_ours=0.016407, max_rel_diff_vs_ours=0.009224, mean_nll=5.635985 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

### samba-infer / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-infer.bytes.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 450.0 | 450.0..450.0 | 1 | - | - | - | 248.4 | - | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 384.6 | 384.6..384.6 | 1 | 1.170 | - | - | 862.4 | - | max_abs_diff_vs_ours=2.176e-06, max_rel_diff_vs_ours=1.223e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 282.7 | 282.7..282.7 | 1 | 1.592 | - | - | 1510.6 | - | max_abs_diff_vs_ours=1.848e-06, max_rel_diff_vs_ours=1.039e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 221.4 | 221.4..221.4 | 1 | 2.033 | - | - | 922.3 | - | max_abs_diff_vs_ours=0.018279, max_rel_diff_vs_ours=0.010276, mean_nll=5.635976 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 182.0 | 182.0..182.0 | 1 | 2.473 | - | - | 1521.3 | - | max_abs_diff_vs_ours=0.016166, max_rel_diff_vs_ours=0.009088, mean_nll=5.635914 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-train-step.bytes.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 159.4 | 159.4..159.4 | 1 | - | - | - | 1676.1 | 1156.0 | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 57.5 | 57.5..57.5 | 1 | 2.771 | - | - | 2932.3 | 311.2 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-compile-tf32 | torch | gpu | opponent | 43.6 | 43.6..43.6 | 1 | 3.655 | - | - | 2371.0 | 316.8 | loss_first_abs_diff_vs_ours=3.815e-05, loss_first_step=5.635948, loss_last_abs_diff_vs_ours=0.0001855, loss_last_step=4.833748, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-bf16 | torch | gpu | opponent | 157.5 | 157.5..157.5 | 1 | 1.012 | - | - | 1519.5 | 330.6 | loss_first_abs_diff_vs_ours=4.196e-05, loss_first_step=5.635952, loss_last_abs_diff_vs_ours=3.338e-05, loss_last_step=4.833967, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-fp32 | torch | gpu | opponent | 134.2 | 134.2..134.2 | 1 | 1.187 | - | - | 1400.9 | 382.4 | loss_first_abs_diff_vs_ours=4.768e-07, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |
| torch-eager-tf32 | torch | gpu | opponent | 147.6 | 147.6..147.6 | 1 | 1.080 | - | - | 1394.7 | 382.4 | loss_first_abs_diff_vs_ours=3.815e-05, loss_first_step=5.635948, loss_last_abs_diff_vs_ours=0.0003433, loss_last_step=4.833591, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T18:37:39Z on 5785dd7f2fd2, gpu (NVIDIA L40S))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-eager-fp32, torch-eager-tf32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| amsgrad | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] |
| dropout | 0.0 | 0.0 |
| eps | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 |
| weight_decay | 0.01 | 0.01 |

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-forward.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6.5 | 6.5..6.5 | 1 | - | - | - | 1507.1 | 900.0 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 3.8 | 3.8..3.8 | 1 | 1.735 | - | - | 1097.7 | 78.4 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-tf32 | torch | gpu | opponent | 3.8 | 3.8..3.8 | 1 | 1.716 | - | - | 1092.7 | 78.4 | max_abs_diff_vs_ours=0.0002124, max_rel_diff_vs_ours=4.545e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 4.7 | 4.7..4.7 | 1 | 1.394 | - | - | 994.8 | 46.9 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-tf32 | torch | gpu | opponent | 3.4 | 3.4..3.4 | 1 | 1.893 | - | - | 946.1 | 46.9 | max_abs_diff_vs_ours=0.0002124, max_rel_diff_vs_ours=4.545e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 3.6 | 3.6..3.6 | 1 | 1.816 | - | - | 1287.2 | 80.1 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.3 | 4.3..4.3 | 1 | 1.523 | - | - | 1228.9 | 41.8 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, torch-eager-fp32, torch-eager-tf32, torch-compile-fp32, torch-compile-tf32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-compile-tf32 | torch-eager-bf16 | torch-eager-fp32 | torch-eager-tf32 |
|---|---||---|---||---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 | 7 | 7 |

### transformer-infer / gaussian (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-infer.gaussian.shape-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 78.8 | 78.8..78.8 | 1 | - | - | - | 144.4 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 93.2 | 93.2..93.2 | 1 | 0.846 | - | - | 727.6 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.076e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 7.8 | 7.8..7.8 | 1 | 10.046 | - | - | 895.6 | - | max_abs_diff_vs_ours=9.537e-07, max_rel_diff_vs_ours=2.153e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 8.0 | 8.0..8.0 | 1 | 9.804 | - | - | 740.1 | - | max_abs_diff_vs_ours=0.001941, max_rel_diff_vs_ours=0.0004383 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 64.1 | 64.1..64.1 | 1 | 1.229 | - | - | 911.2 | - | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0004107 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

## Algorithm expansion

### ard / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.ard.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15000.6 | 15000.6..15000.6 | 1 | - | - | - | 1213.0 | 686.0 | finite=True, r2=-0.122912, rmse=0.885183 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7400.8 | 7400.8..7400.8 | 1 | 2.027 | - | - | 1381.8 | - | finite=True, r2=0.327402, rmse=0.685075 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true |
| max_iter | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.1 | 5.1..5.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.1 | 6.1..6.1 | 1 | 0.845 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ard / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.ard.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 32.7 | 32.7..32.7 | 1 | - | - | - | 256.8 | 686.0 | finite=True, r2=0.909193, rmse=4.799513 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 27.2 | 27.2..27.2 | 1 | 1.203 | - | - | 262.9 | - | finite=True, r2=0.909190, rmse=4.799574 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true |
| max_iter | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.5 | 0.5..0.5 | 1 | 0.795 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bayesian-ridge.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 70103.2 | 70103.2..70103.2 | 1 | - | - | - | 1128.8 | 1454.0 | finite=True, r2=-41643.670747, rmse=170.466932 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8161.0 | 8161.0..8161.0 | 1 | 8.590 | - | - | 3653.3 | - | finite=True, r2=-395441.818968, rmse=525.293799 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true |
| max_iter | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.9 | 5.9..5.9 | 1 | 0.940 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bayesian-ridge.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 314.8 | 314.8..314.8 | 1 | - | - | - | 251.2 | 686.0 | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 210.6 | 210.6..210.6 | 1 | 1.495 | - | - | 391.4 | - | finite=True, r2=0.908983, rmse=4.805056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true |
| max_iter | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.0 | 1.0..1.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.4 | 2.4..2.4 | 1 | 0.416 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.enet-cv.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 73680.5 | 73680.5..73680.5 | 1 | - | - | - | 1237.8 | 1454.0 | finite=True, r2=0.316583, rmse=0.690563 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 112446.5 | 112446.5..112446.5 | 1 | 0.655 | - | - | 8909.5 | - | finite=True, r2=0.326805, rmse=0.685379 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 |
| eps | 0.001 | 0.001 |
| fit_intercept | true | true |
| l1_ratio | 0.5 | 0.5 |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | "auto" | "auto" |
| seed | 7 | 7 |
| selection | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 24.5 | 24.5..24.5 | 1 | 0.224 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.enet-cv.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2309.6 | 2309.6..2309.6 | 1 | - | - | - | 357.7 | 686.0 | finite=True, r2=0.909002, rmse=4.804540 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 507.5 | 507.5..507.5 | 1 | 4.551 | - | - | 645.0 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 |
| eps | 0.001 | 0.001 |
| fit_intercept | true | true |
| l1_ratio | 0.5 | 0.5 |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | "auto" | "auto" |
| seed | 7 | 7 |
| selection | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.8 | 0.8..0.8 | 1 | 1.615 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### gamma / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gamma.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.048e+06 | 1.048e+06..1.048e+06 | 1 | - | - | - | 1171.1 | 1454.0 | finite=True, r2=0.027316, rmse=0.823846 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 157726.8 | 157726.8..157726.8 | 1 | 6.643 | - | - | 2813.1 | - | finite=True, r2=0.181278, rmse=0.755838 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.3 | 5.3..5.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 33.2 | 33.2..33.2 | 1 | 0.159 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### gamma / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.gamma.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 860.8 | 860.8..860.8 | 1 | - | - | - | 296.9 | 686.0 | finite=True, r2=-232.449719, rmse=243.351187 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1738.8 | 1738.8..1738.8 | 1 | 0.495 | - | - | 332.4 | - | finite=True, r2=-232.959247, rmse=243.616612 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.2 | 4.2..4.2 | 1 | 0.112 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### huber / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.huber.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 35732.0 | 35732.0..35732.0 | 1 | - | - | - | 1228.8 | 1454.0 | finite=True, r2=-0.009166, rmse=0.839154 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 89054.9 | 89054.9..89054.9 | 1 | 0.401 | - | - | 3704.6 | - | finite=True, r2=-0.010176, rmse=0.839574 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'epsilon': 1.35, 'fit_intercept': True, 'max_iter': 100, 'tol': 1e-05}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| epsilon | 1.35 | 1.35 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-05 | 1e-05 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.2 | 5.2..5.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 41.6 | 41.6..41.6 | 1 | 0.125 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### huber / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.huber.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8069.6 | 8069.6..8069.6 | 1 | - | - | - | 350.6 | 686.0 | finite=True, r2=0.900215, rmse=5.031176 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4318.6 | 4318.6..4318.6 | 1 | 1.869 | - | - | 459.1 | - | finite=True, r2=0.900215, rmse=5.031163 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'epsilon': 1.35, 'fit_intercept': True, 'max_iter': 100, 'tol': 1e-05}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| epsilon | 1.35 | 1.35 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-05 | 1e-05 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 20.3 | 20.3..20.3 | 1 | 0.070 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15435.9 | 15435.9..15435.9 | 1 | - | - | - | 1128.7 | 1454.0 | finite=True, r2=-1.360350, rmse=1.283360 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 922.6 | 922.6..922.6 | 1 | 16.730 | - | - | 1975.3 | - | finite=True, r2=-4.245e+13, rmse=5.442e+06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 64.2 | 64.2..64.2 | 1 | 240.263 | - | - | 2966.9 | 1374.0 | finite=True, r2=0.328088, rmse=0.684726 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| precompute | "auto" | "auto" | "auto" |
| seed | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.5 | 5.5..5.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.9 | 4.9..4.9 | 1 | 1.109 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | - | 11.5 | 11.5..11.5 | 1 | 0.474 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

inference call, cuml-gpu: predict(Xq)(Xq)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 155.5 | 155.5..155.5 | 1 | - | - | - | 251.5 | 686.0 | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 50.8 | 50.8..50.8 | 1 | 3.063 | - | - | 292.2 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 5.5 | 5.5..5.5 | 1 | 28.460 | - | - | 1043.5 | 498.0 | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| precompute | "auto" | "auto" | "auto" |
| seed | "none (deterministic)" | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.5 | 0.5..0.5 | 1 | 0.902 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | - | 0.6 | 0.6..0.6 | 1 | 0.776 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

inference call, cuml-gpu: predict(Xq)(Xq)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-cv.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 72001.5 | 72001.5..72001.5 | 1 | - | - | - | 1234.0 | 1454.0 | finite=True, r2=0.310329, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 97412.0 | 97412.0..97412.0 | 1 | 0.739 | - | - | 8911.4 | - | finite=True, r2=0.325507, rmse=0.686040 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 |
| eps | 0.001 | 0.001 |
| fit_intercept | true | true |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | "auto" | "auto" |
| seed | 7 | 7 |
| selection | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.4 | 5.4..5.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 8.9 | 8.9..8.9 | 1 | 0.609 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-cv.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2310.1 | 2310.1..2310.1 | 1 | - | - | - | 358.9 | 686.0 | finite=True, r2=0.909059, rmse=4.803051 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 313.1 | 313.1..313.1 | 1 | 7.379 | - | - | 610.4 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 |
| eps | 0.001 | 0.001 |
| fit_intercept | true | true |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | "auto" | "auto" |
| seed | 7 | 7 |
| selection | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.7 | 3.7..3.7 | 1 | 0.345 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-lars.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14760.2 | 14760.2..14760.2 | 1 | - | - | - | 1128.4 | 1454.0 | finite=True, r2=0.310330, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 904.7 | 904.7..904.7 | 1 | 16.314 | - | - | 1969.5 | - | finite=True, r2=0.310816, rmse=0.693470 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true |
| max_iter | 500 | 500 |
| positive | false | false |
| precompute | "auto" | "auto" |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.4 | 5.4..5.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.7 | 4.7..4.7 | 1 | 1.155 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-lars.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 135.4 | 135.4..135.4 | 1 | - | - | - | 251.5 | 686.0 | finite=True, r2=0.908996, rmse=4.804699 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 62.7 | 62.7..62.7 | 1 | 2.160 | - | - | 296.3 | - | finite=True, r2=0.908998, rmse=4.804667 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true |
| max_iter | 500 | 500 |
| positive | false | false |
| precompute | "auto" | "auto" |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 1.050 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-clf.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 135361.3 | 135361.3..135361.3 | 1 | - | - | - | 1184.8 | 1454.0 | accuracy=0.903700 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 11001.2 | 11001.2..11001.2 | 1 | 12.304 | - | - | 1138.3 | - | accuracy=0.890480 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'fit_intercept': True, 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| loss | "hinge" | "hinge" |
| max_iter | 20 | 20 |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 16.4 | 16.4..16.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.1 | 6.1..6.1 | 1 | 2.692 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-clf.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 27089.2 | 27089.2..27089.2 | 1 | - | - | - | 310.3 | 686.0 | accuracy=0.583830 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2681.0 | 2681.0..2681.0 | 1 | 10.104 | - | - | 265.7 | - | accuracy=0.744740 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'fit_intercept': True, 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| loss | "hinge" | "hinge" |
| max_iter | 20 | 20 |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.4 | 13.4..13.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.8 | 1.8..1.8 | 1 | 7.339 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-reg.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 135614.6 | 135614.6..135614.6 | 1 | - | - | - | 1128.5 | 1454.0 | finite=True, r2=-0.282312, rmse=0.945926 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13649.9 | 13649.9..13649.9 | 1 | 9.935 | - | - | 1133.9 | - | finite=True, r2=-0.128155, rmse=0.887248 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.1 | 0.1 |
| fit_intercept | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 20 | 20 |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.4 | 5.4..5.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.8 | 5.8..5.8 | 1 | 0.932 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-reg.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26018.8 | 26018.8..26018.8 | 1 | - | - | - | 251.2 | 686.0 | finite=True, r2=0.853920, rmse=6.087406 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2979.5 | 2979.5..2979.5 | 1 | 8.733 | - | - | 258.6 | - | finite=True, r2=0.795107, rmse=7.209421 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.1 | 0.1 |
| fit_intercept | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 20 | 20 |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.2 | 2.2..2.2 | 1 | 0.741 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### perceptron / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.perceptron.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 121296.9 | 121296.9..121296.9 | 1 | - | - | - | 1184.5 | 1454.0 | accuracy=0.882480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7226.2 | 7226.2..7226.2 | 1 | 16.786 | - | - | 1139.5 | - | accuracy=0.896130 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'early_stopping': False, 'eta0': 1.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| class_weight | null | null |
| eta0 | 1.0 | 1.0 |
| fit_intercept | true | true |
| l1_ratio | 0.15 | 0.15 |
| max_iter | 20 | 20 |
| penalty | null | null |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu penalty: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 16.6 | 16.6..16.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.9 | 6.9..6.9 | 1 | 2.419 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### perceptron / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.perceptron.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25202.0 | 25202.0..25202.0 | 1 | - | - | - | 307.1 | 686.0 | accuracy=0.465380 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1887.7 | 1887.7..1887.7 | 1 | 13.350 | - | - | 261.1 | - | accuracy=0.750520 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'early_stopping': False, 'eta0': 1.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| class_weight | null | null |
| eta0 | 1.0 | 1.0 |
| fit_intercept | true | true |
| l1_ratio | 0.15 | 0.15 |
| max_iter | 20 | 20 |
| penalty | null | null |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu penalty: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 14.5 | 14.5..14.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.9 | 0.9..0.9 | 1 | 16.445 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### poisson / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poisson.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 896843.0 | 896843.0..896843.0 | 1 | - | - | - | 1172.1 | 1454.0 | finite=True, r2=0.340827, rmse=0.678204 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 161169.1 | 161169.1..161169.1 | 1 | 5.565 | - | - | 2789.6 | - | finite=True, r2=0.255075, rmse=0.720969 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.4 | 5.4..5.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 33.1 | 33.1..33.1 | 1 | 0.163 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### poisson / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poisson.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 47566.6 | 47566.6..47566.6 | 1 | - | - | - | 288.8 | 686.0 | finite=True, r2=0.035965, rmse=15.638069 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2143.8 | 2143.8..2143.8 | 1 | 22.188 | - | - | 349.6 | - | finite=True, r2=0.036205, rmse=15.636127 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.4 | 3.4..3.4 | 1 | 0.391 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 153754.8 | 153754.8..153754.8 | 1 | - | - | - | 1213.6 | 686.0 | finite=False, r2=nan, rmse=nan | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.404e+06 | 1.404e+06..1.404e+06 | 1 | 0.109 | - | - | 7995.7 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 5000 | - |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "highs" | "highs" |
| tol | 0.0001 | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.1 | 5.1..5.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 21.7 | 21.7..21.7 | 1 | 0.237 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 35571.7 | 35571.7..35571.7 | 1 | - | - | - | 256.3 | 686.0 | finite=True, r2=0.900039, rmse=5.035615 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 386298.7 | 386298.7..386298.7 | 1 | 0.092 | - | - | 926.3 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 5000 | - |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "highs" | "highs" |
| tol | 0.0001 | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.3 | 4.3..4.3 | 1 | 0.373 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-clf.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14854.5 | 14854.5..14854.5 | 1 | - | - | - | 1190.0 | 1454.0 | accuracy=0.894330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 11222.3 | 11222.3..11222.3 | 1 | 1.324 | - | - | 9513.3 | - | accuracy=0.910540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_intercept': True, 'positive': False, 'random_state': 7, 'solver': 'auto', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| max_iter | null | null |
| positive | false | false |
| seed | 7 | 7 |
| solver | "auto" | "auto" |
| tol | 0.0001 | 0.0001 |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_iter: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 18.0 | 18.0..18.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.7 | 6.7..6.7 | 1 | 2.700 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-clf.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 361.0 | 361.0..361.0 | 1 | - | - | - | 311.3 | 686.0 | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 129.0 | 129.0..129.0 | 1 | 2.800 | - | - | 388.3 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_intercept': True, 'positive': False, 'random_state': 7, 'solver': 'auto', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| max_iter | null | null |
| positive | false | false |
| seed | 7 | 7 |
| solver | "auto" | "auto" |
| tol | 0.0001 | 0.0001 |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_iter: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.0 | 13.0..13.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.7 | 0.7..0.7 | 1 | 17.751 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-cv.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 188507.2 | 188507.2..188507.2 | 1 | - | - | - | 8535.6 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 |
| fit_intercept | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | Xq | - | 6.0 | 6.0..6.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-cv.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 2252.8 | 2252.8..2252.8 | 1 | - | - | - | 378.4 | - | finite=True, r2=0.908983, rmse=4.805055 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 |
| fit_intercept | true | true |
| seed | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 614779.9 | 614779.9..614779.9 | 1 | - | - | - | 1186.9 | 1454.0 | accuracy=0.901150 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 36456.8 | 36456.8..36456.8 | 1 | 16.863 | - | - | 1138.6 | - | accuracy=0.910200 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 4933.9 | 4933.9..4933.9 | 1 | 124.604 | - | - | 3014.7 | 1374.0 | accuracy=0.809350 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | - | - |
| class_weight | - | null | null |
| epochs | 100 | - | - |
| epsilon | - | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | - | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.0 | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: cuml-gpu tol: cuML MBSGD tol=0.0 is its no-early-stop value; ours and scikit-learn tol=None

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 18.4 | 18.4..18.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.7 | 6.7..6.7 | 1 | 2.736 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | - | 2.8 | 2.8..2.8 | 1 | 6.628 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

inference call, cuml-gpu: predict(Xq)(Xq)

### sgd-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 127982.1 | 127982.1..127982.1 | 1 | - | - | - | 308.6 | 686.0 | accuracy=0.766410 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14132.8 | 14132.8..14132.8 | 1 | 9.056 | - | - | 263.2 | - | accuracy=0.752520 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 1642.8 | 1642.8..1642.8 | 1 | 77.906 | - | - | 1087.7 | 498.0 | accuracy=0.703120 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | - | - |
| class_weight | - | null | null |
| epochs | 100 | - | - |
| epsilon | - | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | - | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.0 | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: cuml-gpu tol: cuML MBSGD tol=0.0 is its no-early-stop value; ours and scikit-learn tol=None

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 12.7 | 12.7..12.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.6 | 2.6..2.6 | 1 | 4.904 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | - | 0.8 | 0.8..0.8 | 1 | 16.167 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

inference call, cuml-gpu: predict(Xq)(Xq)

### sgd-ocsvm / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 124050.8 | 124050.8..124050.8 | 1 | - | - | - | 1135.2 | 1454.0 | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7197.0 | 7197.0..7197.0 | 1 | 17.236 | - | - | 1141.9 | - | fraction_flagged=0.093340, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'average': False, 'eta0': 0.0, 'fit_intercept': True, 'learning_rate': 'optimal', 'max_iter': 20, 'nu': 0.1, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| eta0 | 0.0 | 0.0 |
| fit_intercept | true | true |
| learning_rate | "optimal" | "optimal" |
| max_iter | 20 | 20 |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.0 | 13.0..13.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.4 | 5.4..5.4 | 1 | 2.429 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-ocsvm / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26835.2 | 26835.2..26835.2 | 1 | - | - | - | 258.4 | 686.0 | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2009.6 | 2009.6..2009.6 | 1 | 13.354 | - | - | 264.6 | - | fraction_flagged=0.007020, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'average': False, 'eta0': 0.0, 'fit_intercept': True, 'learning_rate': 'optimal', 'max_iter': 20, 'nu': 0.1, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| eta0 | 0.0 | 0.0 |
| fit_intercept | true | true |
| learning_rate | "optimal" | "optimal" |
| max_iter | 20 | 20 |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 9.4 | 9.4..9.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.9 | 0.9..0.9 | 1 | 10.672 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 629928.9 | 629928.9..629928.9 | 1 | - | - | - | 1128.3 | 1454.0 | finite=True, r2=-3.459e+24, rmse=1.554e+12 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 55283.2 | 55283.2..55283.2 | 1 | 11.395 | - | - | 1133.1 | - | finite=True, r2=-2.197e+24, rmse=1.238e+12 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 4959.8 | 4959.8..4959.8 | 1 | 127.006 | - | - | 2924.6 | 1374.0 | finite=True, r2=0.327768, rmse=0.684889 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | - | - |
| epochs | 100 | - | - |
| epsilon | - | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "squared_loss" | "squared_error" | "squared_error" |
| max_iter | - | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.0 | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: cuml-gpu loss: cuML spells squared error 'squared_loss'

accepted difference: cuml-gpu tol: cuML MBSGD tol=0.0 is its no-early-stop value; ours and scikit-learn tol=None

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 6.0 | 6.0..6.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 7.5 | 7.5..7.5 | 1 | 0.796 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | - | 2.4 | 2.4..2.4 | 1 | 2.475 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

inference call, cuml-gpu: predict(Xq)(Xq)

### sgd-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 127806.2 | 127806.2..127806.2 | 1 | - | - | - | 252.0 | 686.0 | finite=True, r2=0.868127, rmse=5.783813 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13336.4 | 13336.4..13336.4 | 1 | 9.583 | - | - | 259.8 | - | finite=True, r2=0.880681, rmse=5.501638 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| cuml-gpu | cuml | gpu | opponent | 1678.8 | 1678.8..1678.8 | 1 | 76.129 | - | - | 968.0 | 498.0 | finite=True, r2=0.908979, rmse=4.805168 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, cuml-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | cuml-gpu | ours | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | cuml (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| batch_size | 4096 | - | - |
| epochs | 100 | - | - |
| epsilon | - | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "squared_loss" | "squared_error" | "squared_error" |
| max_iter | - | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.0 | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: cuml-gpu loss: cuML spells squared error 'squared_loss'

accepted difference: cuml-gpu tol: cuML MBSGD tol=0.0 is its no-early-stop value; ours and scikit-learn tol=None

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.8 | 2.8..2.8 | 1 | 0.580 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| cuml-gpu | Xq | - | 0.6 | 0.6..0.6 | 1 | 2.931 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

inference call, cuml-gpu: predict(Xq)(Xq)

### tweedie / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.tweedie.istella.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 891396.4 | 891396.4..891396.4 | 1 | - | - | - | 1166.7 | 1454.0 | finite=True, r2=-0.089414, rmse=0.871880 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 160871.2 | 160871.2..160871.2 | 1 | 5.541 | - | - | 2783.6 | - | finite=True, r2=-21.706369, rmse=3.980469 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'link': 'log', 'max_iter': 100, 'power': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 5.1 | 5.1..5.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 29.6 | 29.6..29.6 | 1 | 0.174 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### tweedie / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.tweedie.taxi.rows-full.log`, ran on 5785dd7f2fd2

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 630.6 | 630.6..630.6 | 1 | - | - | - | 293.3 | 686.0 | finite=True, r2=-10.070851, rmse=52.994073 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1608.1 | 1608.1..1608.1 | 1 | 0.392 | - | - | 355.8 | - | finite=True, r2=-10.073629, rmse=53.000721 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU nvidia-smi --query-compute-apps used_memory for this pid at the round's end (context and pools; not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'link': 'log', 'max_iter': 100, 'power': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| fit_intercept | true | true |
| max_iter | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.5 | 0.5..0.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.3 | 3.3..3.3 | 1 | 0.145 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML GaussianMixture, GaussianProcessRegressor/Classifier, Nystroem, RBFSampler: cuML 26.8.0 has none; scikit-learn on the CPU is the arm.
- Classical, wave 2, not planned on this vendor: faiss-gpu: no pinned PyPI wheel for this image; cuVS ivf_flat from the pinned rapids set (cuvs-cu12==26.8.1) is the IVF-Flat GPU arm.
- Classical, wave 2, not planned on this vendor: umap-learn and faiss-cpu: not installed on NVIDIA; cuML UMAP and cuVS are the arms.
- Classical, wave 2, not planned on this vendor: cuML SpectralClustering/SpectralEmbedding: present only in newer cuML; if the pinned 26.8.0 lacks them the arm refuses by name and scikit-learn stands beside it.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU tier: `--no-cpu-arm` was passed, so no `ours-cpu` arm ran.
- Our CPU tier, no ours-cpu arm: neural lm-host-train-step, lm-infer, mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: its `ours` arm already IS the CPU path (the public *Inference class runs on the host binding).
- Our CPU tier: a GBDT configuration the host side does not restate refuses by name in its ours-cpu cell (python/mojolearn/host_surface.py NO_CPU_PATH lists them), and a FAST-only run (`--modes fast`) has no ours-cpu arm: the host bindings build IDENTICAL only.
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba torch-* arms are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not deployment kernels. On NVIDIA the mamba-ssm-* arms are mamba_ssm's own fused CUDA/Triton kernels (the deployment path); on AMD and Apple the Mamba lanes race the references only (NOT_PLANNED says why).
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its TF32 setting); torch's bf16 arms carry the lower precision

