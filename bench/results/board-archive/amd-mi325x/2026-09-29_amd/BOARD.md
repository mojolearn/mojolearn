# mojolearn benchmark board

Generated 2026-09-30T04:46:07Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | amd / hip |
| GPU | AMD Instinct Mi325X VF |
| GPU driver | 6.12.12 |
| CPU | AMD EPYC 9575F 64-Core Processor (20 logical cores) |
| memory bytes | 168790056960 |
| OS | Ubuntu 24.04.2 LTS |
| Python | 3.12.3 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-manylinux_2_35_x86_64.whl, sha256 bafc6196927055d47f9113bf90ed378fdcf895ce215aceafa24e8787cd974641) |
| script commit | 4f3e685348ddc41ec22ce00c53187c3528d929b7 |
| patch sync | - |
| modes | identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, torch 2.6.0+rocm6.4.1.git1ded221d, umap-learn 0.5.12, pynndescent 0.6.0, numba 0.67.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.5.3 |

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

Races: 353 planned, 96 done, 1 failed, 256 pending. Cells: 263 (HOST-MEMORY 1, REFUSED 17, ok 245).

Inference cells: 174 (REFUSED 18, ok 156).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | poisson | taxi | r2 (higher is better) | - | 0.035965 | - | sklearn-cpu 0.036205 |
| algos | poisson | taxi | rmse (lower is better) | - | 15.638069 | - | sklearn-cpu 15.636127 |
| algos | sgd-clf | istella | accuracy (higher is better) | - | 0.901150 | - | sklearn-cpu 0.910200 |
| algos | sgd-clf | taxi | accuracy (higher is better) | - | 0.766410 | - | sklearn-cpu 0.752520 |
| algos | sgd-reg | istella | r2 (higher is better) | - | -3.459e+24 | - | sklearn-cpu -2.197e+24 |
| algos | sgd-reg | istella | rmse (lower is better) | - | 1.554e+12 | - | sklearn-cpu 1.238e+12 |
| algos | sgd-reg | taxi | r2 (higher is better) | - | 0.868127 | - | sklearn-cpu 0.880681 |
| algos | sgd-reg | taxi | rmse (lower is better) | - | 5.783813 | - | sklearn-cpu 5.501638 |
| classical | dbscan | istella | n_clusters | - | 40131 | - | sklearn-cpu 40131 |
| classical | dbscan | istella | noise_fraction | - | 0.219391 | - | sklearn-cpu 0.219391 |
| classical | dbscan | istella | rows | - | 1000000 | - | sklearn-cpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | - | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | - | - | - | sklearn-cpu 1.000000 |
| classical | hdbscan | istella | n_clusters | - | - | - | sklearn-cpu 52 |
| classical | hdbscan | istella | noise_fraction | - | - | - | sklearn-cpu 0.252570 |
| classical | hdbscan | istella | rows | - | - | - | sklearn-cpu 100000 |
| classical | hdbscan | taxi | n_clusters | - | - | - | sklearn-cpu 161 |
| classical | hdbscan | taxi | noise_fraction | - | - | - | sklearn-cpu 0.134630 |
| classical | hdbscan | taxi | rows | - | - | - | sklearn-cpu 100000 |
| classical | kde | istella | mean_log_likelihood (higher is better) | - | -222.270586 | - | sklearn-cpu -226.977407 |
| classical | kde | istella | rows_without_density | - | 0 | - | sklearn-cpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | - | -14.826460 | - | sklearn-cpu -14.826437 |
| classical | kde | taxi | rows_without_density | - | 0 | - | sklearn-cpu 0 |
| classical | kmeans | istella | inertia (lower is better) | - | 6.051e+17 | - | sklearn-cpu 6.049e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | inertia_over_ours | - | 1.000000 | - | sklearn-cpu 0.999759; torch-gpu 0.990156 |
| classical | kmeans | istella | n_iter | - | 33 | - | sklearn-cpu 24; torch-gpu 55 |
| classical | kmeans | taxi | inertia (lower is better) | - | 3.093e+08 | - | sklearn-cpu 3.093e+08; torch-gpu 3.129e+08 |
| classical | kmeans | taxi | inertia_over_ours | - | 1.000000 | - | sklearn-cpu 0.999937; torch-gpu 1.011535 |
| classical | kmeans | taxi | n_iter | - | 91 | - | sklearn-cpu 58; torch-gpu 81 |
| classical | knn | istella | recall_at_k (higher is better) | - | 0.976250 | - | sklearn-cpu 1.000000; torch-gpu 0.978680 |
| classical | knn | istella | rows_with_repeated_ids | - | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | - | 0.999754 | - | sklearn-cpu 1.000000; torch-gpu 0.999730 |
| classical | knn | taxi | rows_with_repeated_ids | - | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | - | 0.331944 | - | sklearn-cpu 0.001881; torch-gpu nan; torch-gpu-eigh 0.151604 |
| classical | ols | istella | rmse (lower is better) | - | 0.682027 | - | sklearn-cpu 0.833655; torch-gpu nan; torch-gpu-eigh 0.768589 |
| classical | ols | taxi | r2 (higher is better) | - | 0.908837 | - | sklearn-cpu 0.724850; torch-gpu 0.908840; torch-gpu-eigh 0.908822 |
| classical | ols | taxi | rmse (lower is better) | - | 4.696466 | - | sklearn-cpu 8.159187; torch-gpu 4.696376; torch-gpu-eigh 4.696849 |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | - | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu 1.000000 |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999996 | - | sklearn-cpu 0.999995; torch-gpu 0.999997 |
| classical | svc | istella | accuracy (higher is better) | - | 0.922200 | - | sklearn-cpu 0.922200 |
| classical | svc | istella | n_support | - | 2400 | - | sklearn-cpu 2400 |
| classical | svc | taxi | accuracy (higher is better) | - | 0.767500 | - | sklearn-cpu 0.767500 |
| classical | svc | taxi | n_support | - | 5527 | - | sklearn-cpu 5675 |
| classical2 | agglomerative | istella | n_clusters | - | 8 | - | sklearn-cpu 8 |
| classical2 | agglomerative | istella | silhouette (higher is better) | - | 0.716728 | - | sklearn-cpu 0.716728 |
| classical2 | agglomerative | istella | ari_vs_ours (1 is our partition exactly) | - | - | - | sklearn-cpu 1.000000 |
| classical2 | agglomerative | taxi | n_clusters | - | 8 | - | sklearn-cpu 8 |
| classical2 | agglomerative | taxi | silhouette (higher is better) | - | 0.685524 | - | sklearn-cpu 0.685524 |
| classical2 | agglomerative | taxi | ari_vs_ours (1 is our partition exactly) | - | - | - | sklearn-cpu 1.000000 |
| classical2 | arima | synthetic | forecast_rmse (lower is better) | - | 1.515518 | - | statsmodels-cpu 1.515423 |
| classical2 | arima | synthetic | insample_rmse (lower is better) | - | 0.999342 | - | statsmodels-cpu 0.999338 |
| classical2 | arima | synthetic | mean_aic (lower is better) | - | 5680.976967 | - | statsmodels-cpu 5680.957160 |
| classical2 | arima | synthetic | mean_llf (higher is better) | - | -2836.488483 | - | statsmodels-cpu -2836.478580 |
| classical2 | elasticnet | istella | r2 (higher is better) | - | 0.260922 | - | sklearn-cpu 0.260922 |
| classical2 | elasticnet | istella | rmse (lower is better) | - | 0.718134 | - | sklearn-cpu 0.718134 |
| classical2 | elasticnet | taxi | r2 (higher is better) | - | 0.907378 | - | sklearn-cpu 0.907378 |
| classical2 | elasticnet | taxi | rmse (lower is better) | - | 4.847224 | - | sklearn-cpu 4.847224 |
| classical2 | ets | synthetic | forecast_rmse (lower is better) | - | 0.984392 | - | statsmodels-cpu 0.984664 |
| classical2 | ets | synthetic | insample_rmse (lower is better) | - | 0.990971 | - | statsmodels-cpu 0.992056 |
| classical2 | gmm | istella | bic (lower is better) | - | -3.851e+07 | - | sklearn-cpu -3.901e+07 |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | - | 200.794403 | - | sklearn-cpu 200.768331 |
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
| classical2 | kernel-ridge | istella | r2 (higher is better) | - | 0.385427 | - | sklearn-cpu 0.385427 |
| classical2 | kernel-ridge | istella | rmse (lower is better) | - | 0.646407 | - | sklearn-cpu 0.646407 |
| classical2 | kernel-ridge | taxi | r2 (higher is better) | - | 0.726543 | - | sklearn-cpu 0.726543 |
| classical2 | kernel-ridge | taxi | rmse (lower is better) | - | 8.330373 | - | sklearn-cpu 8.330374 |
| classical2 | knn-clf | istella | accuracy (higher is better) | - | 0.926250 | - | sklearn-cpu 0.926250 |
| classical2 | knn-clf | taxi | accuracy (higher is better) | - | 0.741750 | - | sklearn-cpu 0.741750 |
| classical2 | knn-reg | istella | r2 (higher is better) | - | 0.418145 | - | sklearn-cpu 0.418145 |
| classical2 | knn-reg | istella | rmse (lower is better) | - | 0.625388 | - | sklearn-cpu 0.625388 |
| classical2 | knn-reg | taxi | r2 (higher is better) | - | 0.937323 | - | sklearn-cpu 0.937323 |
| classical2 | knn-reg | taxi | rmse (lower is better) | - | 3.842028 | - | sklearn-cpu 3.842028 |
| classical2 | lasso | istella | r2 (higher is better) | - | 0.310837 | - | sklearn-cpu 0.310837 |
| classical2 | lasso | istella | rmse (lower is better) | - | 0.693460 | - | sklearn-cpu 0.693460 |
| classical2 | lasso | taxi | r2 (higher is better) | - | 0.908995 | - | sklearn-cpu 0.908995 |
| classical2 | lasso | taxi | rmse (lower is better) | - | 4.804745 | - | sklearn-cpu 4.804745 |
| classical2 | linearsvc | istella | accuracy (higher is better) | - | 0.923480 | - | sklearn-cpu 0.923540 |
| classical2 | linearsvc | taxi | accuracy (higher is better) | - | 0.763330 | - | sklearn-cpu 0.763570 |
| classical2 | linearsvr | istella | r2 (higher is better) | - | -0.106754 | - | sklearn-cpu -0.025729 |
| classical2 | linearsvr | istella | rmse (lower is better) | - | 0.878792 | - | sklearn-cpu 0.846012 |
| classical2 | linearsvr | taxi | r2 (higher is better) | - | 0.899813 | - | sklearn-cpu 0.899803 |
| classical2 | linearsvr | taxi | rmse (lower is better) | - | 5.041302 | - | sklearn-cpu 5.041552 |
| classical2 | logreg | istella | accuracy (higher is better) | - | 0.924540 | - | sklearn-cpu 0.924590 |
| classical2 | logreg | istella | logloss (lower is better) | - | 0.181245 | - | sklearn-cpu 0.181264 |
| classical2 | logreg | istella | nonfinite_proba_rows | - | 0 | - | sklearn-cpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | - | 0.763340 | - | sklearn-cpu 0.763320 |
| classical2 | logreg | taxi | logloss (lower is better) | - | 0.538984 | - | sklearn-cpu 0.538980 |
| classical2 | logreg | taxi | nonfinite_proba_rows | - | 0 | - | sklearn-cpu 0 |
| classical2 | nystroem | istella | kernel_rel_error (lower is better) | - | - | - | sklearn-cpu 0.038958 |
| classical2 | nystroem | taxi | kernel_rel_error (lower is better) | - | - | - | sklearn-cpu 0.044369 |
| classical2 | rbf-sampler | istella | kernel_rel_error (lower is better) | - | 0.141980 | - | sklearn-cpu 0.137405 |
| classical2 | rbf-sampler | taxi | kernel_rel_error (lower is better) | - | 0.108549 | - | sklearn-cpu 0.083775 |
| classical2 | ridge | istella | r2 (higher is better) | - | 0.328682 | - | sklearn-cpu 0.328676 |
| classical2 | ridge | istella | rmse (lower is better) | - | 0.684423 | - | sklearn-cpu 0.684426 |
| classical2 | ridge | taxi | r2 (higher is better) | - | 0.908983 | - | sklearn-cpu 0.908983 |
| classical2 | ridge | taxi | rmse (lower is better) | - | 4.805042 | - | sklearn-cpu 4.805056 |
| classical2 | spectral-embedding | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.799386 | - | sklearn-cpu 0.812682 |
| classical2 | spectral-embedding | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.884889 | - | sklearn-cpu 0.898012 |
| classical2 | spectral | istella | n_clusters | - | 8 | - | sklearn-cpu 8 |
| classical2 | spectral | istella | silhouette (higher is better) | - | 0.147668 | - | sklearn-cpu 0.147699 |
| classical2 | spectral | istella | ari_vs_ours (1 is our partition exactly) | - | - | - | sklearn-cpu 0.999826 |
| classical2 | spectral | taxi | n_clusters | - | 8 | - | sklearn-cpu 8 |
| classical2 | spectral | taxi | silhouette (higher is better) | - | 0.039910 | - | sklearn-cpu 0.089894 |
| classical2 | spectral | taxi | ari_vs_ours (1 is our partition exactly) | - | - | - | sklearn-cpu 0.582313 |
| classical2 | svr | istella | r2 (higher is better) | - | 0.318258 | - | sklearn-cpu 0.318248 |
| classical2 | svr | istella | rmse (lower is better) | - | 0.680816 | - | sklearn-cpu 0.680821 |
| classical2 | svr | taxi | r2 (higher is better) | - | 0.767551 | - | sklearn-cpu 0.767550 |
| classical2 | svr | taxi | rmse (lower is better) | - | 7.680395 | - | sklearn-cpu 7.680409 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | - | 0.999992 | - | sklearn-cpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | - | 0.002554 | - | sklearn-cpu 0.000122 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | - | 0.999965 | - | sklearn-cpu 0.999965 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | - | 0.003257 | - | sklearn-cpu 0.003257 |
| classical2 | umap | istella | trustworthiness_k15 (higher is better, 1 at most) | - | 0.979906 | - | umap-learn-cpu 0.977439; umap-learn-cpu-unseeded 0.977474 |
| classical2 | umap | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | 0.990480 | - | umap-learn-cpu 0.990061; umap-learn-cpu-unseeded 0.990396 |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | - | torch-eager-fp32 2.802e-06; torch-compile-fp32 2.802e-06; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 0.001038; torch-compile-fp32 0.001038; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.866e-06; torch-compile-fp32 2.866e-06; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | - | torch-compile-fp32 9.018733; torch-compile-bf16 9.018664; torch-eager-bf16 9.018647; torch-eager-fp32 9.018733 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-compile-fp32 1.311e-06; torch-compile-bf16 0.006870; torch-eager-bf16 0.006042; torch-eager-fp32 1.311e-06 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-compile-fp32 1.8e-06; torch-compile-bf16 0.009432; torch-eager-bf16 0.008295; torch-eager-fp32 1.8e-06 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | - | torch-eager-fp32 9.018732; torch-compile-fp32 9.018733; torch-eager-bf16 9.018646; torch-compile-bf16 9.018663 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.422411 | - | torch-eager-fp32 8.422411; torch-compile-fp32 8.422411; torch-eager-bf16 8.420303; torch-compile-bf16 8.419653 |
| neural | lm-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | lm-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 9.537e-07; torch-compile-fp32 0.000000; torch-eager-bf16 8.678e-05; torch-compile-bf16 6.962e-05 |
| neural | lm-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.002108; torch-compile-bf16 0.002758 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-eager-bf16 5.436e-05 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.189e-07; torch-eager-bf16 2.711e-05 |
| neural | mamba1-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-07; torch-cpu-eager-bf16 5.15e-05 |
| neural | mamba1-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 5.948e-08; torch-cpu-eager-bf16 2.57e-05 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.907e-06; torch-compile-fp32 1.907e-06; torch-eager-bf16 0.007144; torch-compile-bf16 0.007144 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 6.485e-07; torch-compile-fp32 6.485e-07; torch-eager-bf16 0.002429; torch-compile-bf16 0.002429 |
| neural | mamba2-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 7.749e-07; torch-cpu-compile-fp32 7.749e-07; torch-cpu-eager-bf16 0.005920; torch-cpu-compile-bf16 0.005920 |
| neural | mamba2-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.751e-07; torch-cpu-compile-fp32 2.751e-07; torch-cpu-eager-bf16 0.002102; torch-cpu-compile-bf16 0.002102 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 5.364e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.002069; torch-compile-bf16 0.001982 |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.347e-07; torch-compile-fp32 2.086e-07; torch-eager-bf16 0.0009051; torch-compile-bf16 0.0008672 |
| neural | mamba3-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.384e-07; torch-cpu-compile-fp32 2.384e-07; torch-cpu-eager-bf16 0.002148; torch-cpu-compile-bf16 0.002040 |
| neural | mamba3-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.06e-07; torch-cpu-compile-fp32 1.06e-07; torch-cpu-eager-bf16 0.0009547; torch-cpu-compile-bf16 0.0009065 |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-compile-bf16 -; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.001819; torch-eager-fp32 4.768e-07 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-compile-bf16 -; torch-compile-fp32 1.02e-07; torch-eager-bf16 0.0003893; torch-eager-fp32 1.02e-07 |
| neural | transformer-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-bf16 -; torch-cpu-compile-fp32 7.153e-07; torch-cpu-eager-bf16 0.001941; torch-cpu-eager-fp32 4.768e-07 |
| neural | transformer-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-bf16 -; torch-cpu-compile-fp32 1.615e-07; torch-cpu-eager-bf16 0.0004383; torch-cpu-eager-fp32 1.076e-07 |
| trees | et | istella | auc (higher is better) | - | 0.937987 | - | sklearn-et-cpu 0.937659; lightgbm-cpu 0.948128 |
| trees | et | istella | logloss (lower is better) | - | 0.189989 | - | sklearn-et-cpu 0.190177; lightgbm-cpu 0.197270 |
| trees | et | taxi | auc (higher is better) | - | 0.618907 | - | sklearn-et-cpu 0.619262; lightgbm-cpu 0.611172 |
| trees | et | taxi | logloss (lower is better) | - | 0.526142 | - | sklearn-et-cpu 0.525951; lightgbm-cpu 0.529303 |
| trees | gbdt-categorical | taxi | auc (higher is better) | - | 0.630363 | - | catboost-cpu 0.628772; xgboost-gpu -; xgboost-cpu 0.631473; lightgbm-cpu 0.632665 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | - | 0.528463 | - | catboost-cpu 0.528923; xgboost-gpu -; xgboost-cpu 0.528686; lightgbm-cpu 0.528094 |
| trees | gbdt-depthwise | istella | auc (higher is better) | - | 0.979129 | - | catboost-cpu 0.983135; xgboost-gpu -; xgboost-cpu 0.983622 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | - | 0.188483 | - | catboost-cpu 0.157692; xgboost-gpu -; xgboost-cpu 0.149263 |
| trees | gbdt-depthwise | taxi | auc (higher is better) | - | 0.625417 | - | catboost-cpu 0.632578; xgboost-gpu -; xgboost-cpu 0.630968 |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | - | 0.530232 | - | catboost-cpu 0.527851; xgboost-gpu -; xgboost-cpu 0.528677 |
| trees | gbdt-lossguide | istella | auc (higher is better) | - | 0.983749 | - | catboost-cpu 0.983135; xgboost-gpu -; xgboost-cpu 0.983622; lightgbm-cpu 0.983778 |
| trees | gbdt-lossguide | istella | logloss (lower is better) | - | 0.149364 | - | catboost-cpu 0.157692; xgboost-gpu -; xgboost-cpu 0.149263; lightgbm-cpu 0.149653 |
| trees | gbdt-lossguide | taxi | auc (higher is better) | - | 0.631154 | - | catboost-cpu 0.632578; xgboost-gpu -; xgboost-cpu 0.630968; lightgbm-cpu 0.632243 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | - | 0.528317 | - | catboost-cpu 0.527851; xgboost-gpu -; xgboost-cpu 0.528677; lightgbm-cpu 0.528067 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | - | 0.903294 | - | catboost-cpu 0.907768; xgboost-gpu -; xgboost-cpu 0.910140; lightgbm-cpu 0.910058 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | - | 0.281958 | - | catboost-cpu 0.258286; xgboost-gpu -; xgboost-cpu 0.246803; lightgbm-cpu 0.245916 |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | - | 0.596646 | - | catboost-cpu 0.599150; xgboost-gpu -; xgboost-cpu 0.601128; lightgbm-cpu 0.601580 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | - | 1.022664 | - | catboost-cpu 1.012734; xgboost-gpu -; xgboost-cpu 1.005128; lightgbm-cpu 1.004282 |
| trees | gbdt-ordered | istella | auc (higher is better) | - | 0.972482 | - | catboost-cpu 0.979221 |
| trees | gbdt-ordered | istella | logloss (lower is better) | - | 0.227811 | - | catboost-cpu 0.192114 |
| trees | gbdt-ordered | taxi | auc (higher is better) | - | 0.620358 | - | catboost-cpu 0.628918 |
| trees | gbdt-ordered | taxi | logloss (lower is better) | - | 0.531519 | - | catboost-cpu 0.529083 |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | - | 0.844605 | - | catboost-cpu 0.846328; xgboost-gpu -; xgboost-cpu 0.872796 |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | - | 0.711712 | - | catboost-cpu 0.713361; xgboost-gpu -; xgboost-cpu 0.738397 |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | - | 0.641863 | - | catboost-cpu 0.643611; xgboost-gpu -; xgboost-cpu 0.670093 |
| trees | gbdt-rank-yetirank | istella | map (higher is better) | - | 0.814902 | - | catboost-cpu 0.851990; xgboost-gpu -; xgboost-cpu 0.842929; lightgbm-cpu 0.858362 |
| trees | gbdt-rank-yetirank | istella | ndcg10 (higher is better) | - | 0.680993 | - | catboost-cpu 0.726111; xgboost-gpu -; xgboost-cpu 0.726256; lightgbm-cpu 0.741515 |
| trees | gbdt-rank-yetirank | istella | ndcg5 (higher is better) | - | 0.615076 | - | catboost-cpu 0.660263; xgboost-gpu -; xgboost-cpu 0.664249; lightgbm-cpu 0.680254 |
| trees | gbdt-symmetric-1000 | istella | auc (higher is better) | - | 0.977075 | - | catboost-cpu 0.982309 |
| trees | gbdt-symmetric-1000 | istella | logloss (lower is better) | - | 0.203751 | - | catboost-cpu 0.171620 |
| trees | gbdt-symmetric-1000 | taxi | auc (higher is better) | - | 0.621310 | - | catboost-cpu 0.631642 |
| trees | gbdt-symmetric-1000 | taxi | logloss (lower is better) | - | 0.531329 | - | catboost-cpu 0.528267 |
| trees | gbdt-symmetric | istella | auc (higher is better) | - | 0.977075 | - | catboost-cpu 0.979899 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | - | 0.203751 | - | catboost-cpu 0.188093 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | - | 0.621310 | - | catboost-cpu 0.630269 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | - | 0.531329 | - | catboost-cpu 0.528650 |
| trees | iforest | istella | auc (higher is better) | - | 0.830358 | - | sklearn-iforest-cpu 0.827914 |
| trees | iforest | taxi | auc (higher is better) | - | 0.551846 | - | sklearn-iforest-cpu 0.552849 |
| trees | rf | istella | auc (higher is better) | - | 0.945385 | - | sklearn-rf-cpu 0.945303; lightgbm-cpu 0.945361 |
| trees | rf | istella | logloss (lower is better) | - | 0.182017 | - | sklearn-rf-cpu 0.182308; lightgbm-cpu 0.195422 |
| trees | rf | taxi | auc (higher is better) | - | 0.617838 | - | sklearn-rf-cpu 0.617678; lightgbm-cpu 0.617040 |
| trees | rf | taxi | logloss (lower is better) | - | 0.525953 | - | sklearn-rf-cpu 0.525532; lightgbm-cpu 0.526421 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | poisson | taxi | Xq | - | - | 0.7 | - | - | - | sklearn-cpu 1.8 ms (IDENTICAL/arm 0.368) |
| algos | sgd-clf | istella | Xq | - | - | 10.2 | - | - | - | sklearn-cpu 3.7 ms (IDENTICAL/arm 2.741) |
| algos | sgd-clf | taxi | Xq | - | - | 8.2 | - | - | - | sklearn-cpu 0.6 ms (IDENTICAL/arm 12.988) |
| algos | sgd-reg | istella | Xq | - | - | 2.3 | - | - | - | sklearn-cpu 3.9 ms (IDENTICAL/arm 0.580) |
| algos | sgd-reg | taxi | Xq | - | - | 0.7 | - | - | - | sklearn-cpu 0.7 ms (IDENTICAL/arm 0.999) |
| classical | kmeans | istella | Xq | 500000 | - | 41.2 | - | - | - | sklearn-cpu 23.3 ms (IDENTICAL/arm 1.764); torch-gpu 0.8 ms (IDENTICAL/arm 53.873) |
| classical | kmeans | taxi | Xq | 500000 | - | 4.5 | - | - | - | sklearn-cpu 4.6 ms (IDENTICAL/arm 0.974); torch-gpu 0.5 ms (IDENTICAL/arm 9.584) |
| classical | ols | istella | Xq | 500000 | - | 12.3 | - | - | - | sklearn-cpu 18.5 ms (IDENTICAL/arm 0.665); torch-gpu 0.5 ms (IDENTICAL/arm 23.721); torch-gpu-eigh 0.5 ms (IDENTICAL/arm 25.155) |
| classical | ols | taxi | Xq | 500000 | - | 1.6 | - | - | - | sklearn-cpu 1.8 ms (IDENTICAL/arm 0.898); torch-gpu 0.5 ms (IDENTICAL/arm 3.448); torch-gpu-eigh 0.4 ms (IDENTICAL/arm 3.706) |
| classical | pca | istella | Xq | 500000 | - | 17.7 | - | - | - | sklearn-cpu 24.7 ms (IDENTICAL/arm 0.715); torch-gpu 0.6 ms (IDENTICAL/arm 29.422) |
| classical | pca | taxi | Xq | 500000 | - | 6.9 | - | - | - | sklearn-cpu 5.6 ms (IDENTICAL/arm 1.220); torch-gpu 0.3 ms (IDENTICAL/arm 21.310) |
| classical | svc | istella | Xq | 10000 | - | 7.3 | - | - | - | sklearn-cpu 1940.9 ms (IDENTICAL/arm 0.004) |
| classical | svc | taxi | Xq | 10000 | - | 4.0 | - | - | - | sklearn-cpu 1005.9 ms (IDENTICAL/arm 0.004) |
| trees | et | istella | test | 500000 | - | 32.9 | - | - | - | sklearn-et-cpu 192.1 ms (IDENTICAL/arm 0.171); lightgbm-cpu 344.4 ms (IDENTICAL/arm 0.096) |
| trees | et | istella | large | 1000000 | - | 61.9 | - | - | - | sklearn-et-cpu 368.7 ms (IDENTICAL/arm 0.168); lightgbm-cpu 683.6 ms (IDENTICAL/arm 0.091) |
| trees | et | taxi | test | 500000 | - | 7.8 | - | - | - | sklearn-et-cpu 117.0 ms (IDENTICAL/arm 0.067); lightgbm-cpu 125.3 ms (IDENTICAL/arm 0.063) |
| trees | et | taxi | large | 1000000 | - | 13.1 | - | - | - | sklearn-et-cpu 203.3 ms (IDENTICAL/arm 0.064); lightgbm-cpu 252.8 ms (IDENTICAL/arm 0.052) |
| trees | gbdt-categorical | taxi | test | 500000 | - | 150.5 | - | - | - | catboost-cpu 623.0 ms (IDENTICAL/arm 0.242); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 373.3 ms (IDENTICAL/arm 0.403); lightgbm-cpu 912.5 ms (IDENTICAL/arm 0.165) |
| trees | gbdt-categorical | taxi | large | 1000000 | - | 342.8 | - | - | - | catboost-cpu 1231.2 ms (IDENTICAL/arm 0.278); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 748.9 ms (IDENTICAL/arm 0.458); lightgbm-cpu 1790.8 ms (IDENTICAL/arm 0.191) |
| trees | gbdt-depthwise | istella | test | 500000 | - | 110.8 | - | - | - | catboost-cpu 206.1 ms (IDENTICAL/arm 0.538); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 116.0 ms (IDENTICAL/arm 0.955) |
| trees | gbdt-depthwise | istella | large | 1000000 | - | 246.1 | - | - | - | catboost-cpu 407.9 ms (IDENTICAL/arm 0.603); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 244.9 ms (IDENTICAL/arm 1.005) |
| trees | gbdt-depthwise | taxi | test | 500000 | - | 94.8 | - | - | - | catboost-cpu 183.9 ms (IDENTICAL/arm 0.515); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 121.3 ms (IDENTICAL/arm 0.781) |
| trees | gbdt-depthwise | taxi | large | 1000000 | - | 210.4 | - | - | - | catboost-cpu 311.7 ms (IDENTICAL/arm 0.675); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 203.5 ms (IDENTICAL/arm 1.034) |
| trees | gbdt-lossguide | istella | test | 500000 | - | 117.7 | - | - | - | catboost-cpu 230.6 ms (IDENTICAL/arm 0.511); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 121.0 ms (IDENTICAL/arm 0.973); lightgbm-cpu 623.9 ms (IDENTICAL/arm 0.189) |
| trees | gbdt-lossguide | istella | large | 1000000 | - | 261.5 | - | - | - | catboost-cpu 454.3 ms (IDENTICAL/arm 0.575); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 240.9 ms (IDENTICAL/arm 1.085); lightgbm-cpu 1252.3 ms (IDENTICAL/arm 0.209) |
| trees | gbdt-lossguide | taxi | test | 500000 | - | 94.7 | - | - | - | catboost-cpu 155.4 ms (IDENTICAL/arm 0.610); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 116.6 ms (IDENTICAL/arm 0.812); lightgbm-cpu 571.7 ms (IDENTICAL/arm 0.166) |
| trees | gbdt-lossguide | taxi | large | 1000000 | - | 211.9 | - | - | - | catboost-cpu 309.0 ms (IDENTICAL/arm 0.686); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 211.5 ms (IDENTICAL/arm 1.002); lightgbm-cpu 1140.5 ms (IDENTICAL/arm 0.186) |
| trees | gbdt-multiclass | istella | test | 500000 | - | 33.4 | - | - | - | catboost-cpu 117.4 ms (IDENTICAL/arm 0.284); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 560.7 ms (IDENTICAL/arm 0.060); lightgbm-cpu 3682.3 ms (IDENTICAL/arm 0.009) |
| trees | gbdt-multiclass | istella | large | 1000000 | - | 49.3 | - | - | - | catboost-cpu 147.6 ms (IDENTICAL/arm 0.334); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 1114.8 ms (IDENTICAL/arm 0.044); lightgbm-cpu 7295.8 ms (IDENTICAL/arm 0.007) |
| trees | gbdt-multiclass | taxi | test | 500000 | - | 11.6 | - | - | - | catboost-cpu 41.1 ms (IDENTICAL/arm 0.282); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 421.6 ms (IDENTICAL/arm 0.028); lightgbm-cpu 3208.1 ms (IDENTICAL/arm 0.004) |
| trees | gbdt-multiclass | taxi | large | 1000000 | - | 18.6 | - | - | - | catboost-cpu 41.6 ms (IDENTICAL/arm 0.448); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 840.1 ms (IDENTICAL/arm 0.022); lightgbm-cpu 6361.5 ms (IDENTICAL/arm 0.003) |
| trees | gbdt-ordered | istella | test | 500000 | - | 21.9 | - | - | - | catboost-cpu 44.4 ms (IDENTICAL/arm 0.493) |
| trees | gbdt-ordered | istella | large | 1000000 | - | 33.4 | - | - | - | catboost-cpu 118.1 ms (IDENTICAL/arm 0.283) |
| trees | gbdt-ordered | taxi | test | 500000 | - | 3.4 | - | - | - | catboost-cpu 16.5 ms (IDENTICAL/arm 0.203) |
| trees | gbdt-ordered | taxi | large | 1000000 | - | 5.4 | - | - | - | catboost-cpu 42.3 ms (IDENTICAL/arm 0.127) |
| trees | gbdt-rank-pairlogit | istella | test | 681250 | - | 28.4 | - | - | - | catboost-cpu 63.2 ms (IDENTICAL/arm 0.449); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 47.0 ms (IDENTICAL/arm 0.605) |
| trees | gbdt-rank-pairlogit | istella | large | 1000000 | - | 36.4 | - | - | - | catboost-cpu 102.9 ms (IDENTICAL/arm 0.354); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 45.2 ms (IDENTICAL/arm 0.806) |
| trees | gbdt-rank-yetirank | istella | test | 681250 | - | 29.6 | - | - | - | catboost-cpu 48.8 ms (IDENTICAL/arm 0.607); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 29.9 ms (IDENTICAL/arm 0.993); lightgbm-cpu 139.3 ms (IDENTICAL/arm 0.213) |
| trees | gbdt-rank-yetirank | istella | large | 1000000 | - | 36.9 | - | - | - | catboost-cpu 63.3 ms (IDENTICAL/arm 0.584); xgboost-gpu - ms (IDENTICAL/arm -); xgboost-cpu 42.8 ms (IDENTICAL/arm 0.864); lightgbm-cpu 204.6 ms (IDENTICAL/arm 0.181) |
| trees | gbdt-symmetric-1000 | istella | test | 500000 | - | 21.5 | - | - | - | catboost-cpu 88.8 ms (IDENTICAL/arm 0.242) |
| trees | gbdt-symmetric-1000 | istella | large | 1000000 | - | 36.5 | - | - | - | catboost-cpu 108.7 ms (IDENTICAL/arm 0.336) |
| trees | gbdt-symmetric-1000 | taxi | test | 500000 | - | 6.8 | - | - | - | catboost-cpu 24.4 ms (IDENTICAL/arm 0.277) |
| trees | gbdt-symmetric-1000 | taxi | large | 1000000 | - | 12.5 | - | - | - | catboost-cpu 55.0 ms (IDENTICAL/arm 0.228) |
| trees | gbdt-symmetric | istella | test | 500000 | - | 20.2 | - | - | - | catboost-cpu 70.1 ms (IDENTICAL/arm 0.288) |
| trees | gbdt-symmetric | istella | large | 1000000 | - | 39.9 | - | - | - | catboost-cpu 110.8 ms (IDENTICAL/arm 0.360) |
| trees | gbdt-symmetric | taxi | test | 500000 | - | 4.4 | - | - | - | catboost-cpu 16.8 ms (IDENTICAL/arm 0.263) |
| trees | gbdt-symmetric | taxi | large | 1000000 | - | 7.8 | - | - | - | catboost-cpu 42.7 ms (IDENTICAL/arm 0.184) |
| trees | iforest | istella | test | 500000 | - | 571.5 | - | - | - | sklearn-iforest-cpu 1997.8 ms (IDENTICAL/arm 0.286) |
| trees | iforest | istella | large | 1000000 | - | 1010.4 | - | - | - | sklearn-iforest-cpu 3985.8 ms (IDENTICAL/arm 0.254) |
| trees | iforest | taxi | test | 500000 | - | 103.7 | - | - | - | sklearn-iforest-cpu 759.3 ms (IDENTICAL/arm 0.137) |
| trees | iforest | taxi | large | 1000000 | - | 138.1 | - | - | - | sklearn-iforest-cpu 1475.0 ms (IDENTICAL/arm 0.094) |
| trees | rf | istella | test | 500000 | - | 59.8 | - | - | - | sklearn-rf-cpu 637.2 ms (IDENTICAL/arm 0.094); lightgbm-cpu 700.4 ms (IDENTICAL/arm 0.085) |
| trees | rf | istella | large | 1000000 | - | 118.8 | - | - | - | sklearn-rf-cpu 1405.6 ms (IDENTICAL/arm 0.084); lightgbm-cpu 1370.8 ms (IDENTICAL/arm 0.087) |
| trees | rf | taxi | test | 500000 | - | 12.9 | - | - | - | sklearn-rf-cpu 302.6 ms (IDENTICAL/arm 0.043); lightgbm-cpu 704.4 ms (IDENTICAL/arm 0.018) |
| trees | rf | taxi | large | 1000000 | - | 22.6 | - | - | - | sklearn-rf-cpu 561.6 ms (IDENTICAL/arm 0.040); lightgbm-cpu 1404.5 ms (IDENTICAL/arm 0.016) |

## Trees

### et / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/et.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1197.9 | 1197.9..1197.9 | 1 | - | - | - | 40016.5 | - | auc=0.937987, logloss=0.189989 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 18137.9 | 18137.9..18137.9 | 1 | 0.066 | - | - | 42208.4 | - | auc=0.937659, logloss=0.190177 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 77242.5 | 77242.5..77242.5 | 1 | 0.016 | - | - | 50019.4 | - | auc=0.948128, logloss=0.197270 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-et-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,sklearn-et-cpu leaves=lightgbm-cpu:769496,ours:1029236,sklearn-et-cpu:1006311 spread=0.2524 verdict=NOT-COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | sklearn-et-cpu |
|---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - |
| bootstrap | - | false | false |
| class_weight | null | null | null |
| criterion | - | "gini" | "gini" |
| feature_fraction | 1.0 | - | - |
| feature_fraction_bynode | 0.06363636363636363 | - | - |
| learning_rate | 1.0 | - | - |
| max_bin | 255 | - | - |
| max_depth | 16 | 16 | 16 |
| max_features | - | "sqrt" | "sqrt" |
| max_leaves | 65536 | null | null |
| max_samples | - | null | null |
| min_child_weight | 0.0 | - | - |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 |
| reg_alpha | 0.0 | - | - |
| reg_lambda | 0.0 | - | - |
| seed | 7 | 7 | 7 |
| subsample | 0.632 | - | - |

accepted difference: sklearn-et-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-et-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: sklearn-et-cpu max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 32.9 | 32.9..32.9 | 1 | - | - | - | auc=0.937987, auc_matches_fit=True, logloss=0.189989, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 192.1 | 192.1..192.1 | 1 | 0.171 | - | - | auc=0.937659, auc_matches_fit=True, logloss=0.190177, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 344.4 | 344.4..344.4 | 1 | 0.096 | - | - | auc=0.948128, auc_matches_fit=True, logloss=0.197270, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 61.9 | 61.9..61.9 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 368.7 | 368.7..368.7 | 1 | 0.168 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 683.6 | 683.6..683.6 | 1 | 0.091 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### et / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/et.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 889.2 | 889.2..889.2 | 1 | - | - | - | 8660.9 | - | auc=0.618907, logloss=0.526142 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 12543.8 | 12543.8..12543.8 | 1 | 0.071 | - | - | 11451.5 | - | auc=0.619262, logloss=0.525951 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 7161.0 | 7161.0..7161.0 | 1 | 0.124 | - | - | 11258.4 | - | auc=0.611172, logloss=0.529303 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-et-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,sklearn-et-cpu leaves=lightgbm-cpu:66043,ours:881399,sklearn-et-cpu:906974 spread=0.9272 verdict=NOT-COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | sklearn-et-cpu |
|---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - |
| bootstrap | - | false | false |
| class_weight | null | null | null |
| criterion | - | "gini" | "gini" |
| feature_fraction | 1.0 | - | - |
| feature_fraction_bynode | 0.25 | - | - |
| learning_rate | 1.0 | - | - |
| max_bin | 255 | - | - |
| max_depth | 16 | 16 | 16 |
| max_features | - | "sqrt" | "sqrt" |
| max_leaves | 65536 | null | null |
| max_samples | - | null | null |
| min_child_weight | 0.0 | - | - |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 |
| reg_alpha | 0.0 | - | - |
| reg_lambda | 0.0 | - | - |
| seed | 7 | 7 | 7 |
| subsample | 0.632 | - | - |

accepted difference: sklearn-et-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-et-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: sklearn-et-cpu max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 7.8 | 7.8..7.8 | 1 | - | - | - | auc=0.618907, auc_matches_fit=True, logloss=0.526142, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 117.0 | 117.0..117.0 | 1 | 0.067 | - | - | auc=0.619262, auc_matches_fit=True, logloss=0.525951, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 125.3 | 125.3..125.3 | 1 | 0.063 | - | - | auc=0.611172, auc_matches_fit=True, logloss=0.529303, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 13.1 | 13.1..13.1 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 203.3 | 203.3..203.3 | 1 | 0.064 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 252.8 | 252.8..252.8 | 1 | 0.052 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-categorical.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24361.0 | 24361.0..24361.0 | 1 | - | - | - | 9400.3 | - | auc=0.630363, logloss=0.528463 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 97857.4 | 97857.4..97857.4 | 1 | 0.249 | - | - | 11918.4 | - | auc=0.628772, logloss=0.528923 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 18924.8 | 18924.8..18924.8 | 1 | 1.287 | - | - | 8838.5 | - | auc=0.631473, logloss=0.528686 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 10928.0 | 10928.0..10928.0 | 1 | 2.229 | - | - | 8741.9 | - | auc=0.632665, logloss=0.528094 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-categorical arms=catboost-cpu,lightgbm-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:80373,lightgbm-cpu:92031,ours:42048,xgboost-cpu:99609,xgboost-gpu:99609 spread=0.5779 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | - | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | - | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - | - |
| random_strength | 0.0 | - | 0.0 | - | - |
| reg_alpha | - | 0.0 | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | - | "NewtonL2" | - | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 | 1.0 |

accepted difference: catboost-cpu score_function: CatBoost's CPU learner scores splits with Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and LightGBM

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 150.5 | 150.5..150.5 | 1 | - | - | - | auc=0.630363, auc_matches_fit=True, logloss=0.528463, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 623.0 | 623.0..623.0 | 1 | 0.242 | - | - | auc=0.628772, auc_matches_fit=True, logloss=0.528923, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 500000 | 373.3 | 373.3..373.3 | 1 | 0.403 | - | - | auc=0.631473, auc_matches_fit=True, logloss=0.528686, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 912.5 | 912.5..912.5 | 1 | 0.165 | - | - | auc=0.632665, auc_matches_fit=True, logloss=0.528094, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 342.8 | 342.8..342.8 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 1231.2 | 1231.2..1231.2 | 1 | 0.278 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 748.9 | 748.9..748.9 | 1 | 0.458 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1790.8 | 1790.8..1790.8 | 1 | 0.191 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X) on the float32 codes, column 1

inference call, catboost-cpu: catboost predict_proba(int64 categorical frame built in the clock, task_type CPU), column 1

inference call, xgboost-gpu: xgboost XGBClassifier.predict_proba(pandas CategoricalDtype frame built in the clock; inplace_predict takes no category frame here), column 1

inference call, xgboost-cpu: xgboost XGBClassifier.predict_proba(pandas CategoricalDtype frame built in the clock; inplace_predict takes no category frame here), column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) on the float32 codes (its categorical columns are recorded in the model), probability

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6971.8 | 6971.8..6971.8 | 1 | - | - | - | 11771.2 | - | auc=0.979129, logloss=0.188483 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 70785.7 | 70785.7..70785.7 | 1 | 0.098 | - | - | 10524.7 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 16034.8 | 16034.8..16034.8 | 1 | 0.435 | - | - | 11639.5 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:105455,ours:38550,xgboost-cpu:107149,xgboost-gpu:107149 spread=0.6402 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | "No" | - | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 | 1.0 |
| grow_policy | "Depthwise" | "Depthwise" | "depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | - | - |
| leaf_estimation_method | "Newton" | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | - | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | null | 0.0 | 0.0 |
| min_samples_leaf | 1 | 1 | - | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | - | - |
| random_strength | 0.0 | 0.0 | - | - |
| reg_alpha | - | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | 1.0 | 1.0 |

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 110.8 | 110.8..110.8 | 1 | - | - | - | auc=0.979129, auc_matches_fit=True, logloss=0.188483, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 206.1 | 206.1..206.1 | 1 | 0.538 | - | - | auc=0.983135, auc_matches_fit=True, logloss=0.157692, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 500000 | 116.0 | 116.0..116.0 | 1 | 0.955 | - | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 246.1 | 246.1..246.1 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 407.9 | 407.9..407.9 | 1 | 0.603 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 244.9 | 244.9..244.9 | 1 | 1.005 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy here, so XGBoost's own device-mismatch fallback), probability

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6383.5 | 6383.5..6383.5 | 1 | - | - | - | 6545.3 | - | auc=0.625417, logloss=0.530232 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 43791.1 | 43791.1..43791.1 | 1 | 0.146 | - | - | 7279.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 8079.8 | 8079.8..8079.8 | 1 | 0.790 | - | - | 7413.2 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:103827,ours:41324,xgboost-cpu:91234,xgboost-gpu:91234 spread=0.6020 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | "No" | - | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 | 1.0 |
| grow_policy | "Depthwise" | "Depthwise" | "depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | - | - |
| leaf_estimation_method | "Newton" | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | - | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | null | 0.0 | 0.0 |
| min_samples_leaf | 1 | 1 | - | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | - | - |
| random_strength | 0.0 | 0.0 | - | - |
| reg_alpha | - | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | 1.0 | 1.0 |

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 94.8 | 94.8..94.8 | 1 | - | - | - | auc=0.625417, auc_matches_fit=True, logloss=0.530232, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 183.9 | 183.9..183.9 | 1 | 0.515 | - | - | auc=0.632578, auc_matches_fit=True, logloss=0.527851, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 500000 | 121.3 | 121.3..121.3 | 1 | 0.781 | - | - | auc=0.630968, auc_matches_fit=True, logloss=0.528677, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 210.4 | 210.4..210.4 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 311.7 | 311.7..311.7 | 1 | 0.675 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 203.5 | 203.5..203.5 | 1 | 1.034 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy here, so XGBoost's own device-mismatch fallback), probability

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25520.2 | 25520.2..25520.2 | 1 | - | - | - | 11828.3 | - | auc=0.983749, logloss=0.149364 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 103179.7 | 103179.7..103179.7 | 1 | 0.247 | - | - | 10528.0 | - | auc=0.983135, logloss=0.157692 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 23613.1 | 23613.1..23613.1 | 1 | 1.081 | - | - | 11234.1 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 19843.3 | 19843.3..19843.3 | 1 | 1.286 | - | - | 11308.2 | - | auc=0.983778, logloss=0.149653 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:105455,lightgbm-cpu:94845,ours:113893,xgboost-cpu:107149,xgboost-gpu:107149 spread=0.1672 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | - | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | - | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - | - |
| random_strength | 0.0 | - | 0.0 | - | - |
| reg_alpha | - | 0.0 | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | - | "NewtonL2" | - | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 | 1.0 |

accepted difference: catboost-cpu score_function: CatBoost's CPU learner scores splits with Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and LightGBM

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 117.7 | 117.7..117.7 | 1 | - | - | - | auc=0.983749, auc_matches_fit=True, logloss=0.149364, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 230.6 | 230.6..230.6 | 1 | 0.511 | - | - | auc=0.983135, auc_matches_fit=True, logloss=0.157692, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 500000 | 121.0 | 121.0..121.0 | 1 | 0.973 | - | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 623.9 | 623.9..623.9 | 1 | 0.189 | - | - | auc=0.983778, auc_matches_fit=True, logloss=0.149653, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 261.5 | 261.5..261.5 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 454.3 | 454.3..454.3 | 1 | 0.575 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 240.9 | 240.9..240.9 | 1 | 1.085 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1252.3 | 1252.3..1252.3 | 1 | 0.209 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy here, so XGBoost's own device-mismatch fallback), probability

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12994.6 | 12994.6..12994.6 | 1 | - | - | - | 6313.4 | - | auc=0.631154, logloss=0.528317 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 77432.0 | 77432.0..77432.0 | 1 | 0.168 | - | - | 6674.6 | - | auc=0.632578, logloss=0.527851 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 10350.4 | 10350.4..10350.4 | 1 | 1.255 | - | - | 6853.6 | - | auc=0.630968, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 10099.2 | 10099.2..10099.2 | 1 | 1.287 | - | - | 6933.8 | - | auc=0.632243, logloss=0.528067 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:103827,lightgbm-cpu:79876,ours:56739,xgboost-cpu:91234,xgboost-gpu:91234 spread=0.4535 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | - | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | - | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - | - |
| random_strength | 0.0 | - | 0.0 | - | - |
| reg_alpha | - | 0.0 | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | - | "NewtonL2" | - | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 | 1.0 |

accepted difference: catboost-cpu score_function: CatBoost's CPU learner scores splits with Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and LightGBM

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 94.7 | 94.7..94.7 | 1 | - | - | - | auc=0.631154, auc_matches_fit=True, logloss=0.528317, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 155.4 | 155.4..155.4 | 1 | 0.610 | - | - | auc=0.632578, auc_matches_fit=True, logloss=0.527851, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 500000 | 116.6 | 116.6..116.6 | 1 | 0.812 | - | - | auc=0.630968, auc_matches_fit=True, logloss=0.528677, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 571.7 | 571.7..571.7 | 1 | 0.166 | - | - | auc=0.632243, auc_matches_fit=True, logloss=0.528067, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 211.9 | 211.9..211.9 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 309.0 | 309.0..309.0 | 1 | 0.686 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 211.5 | 211.5..211.5 | 1 | 1.002 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1140.5 | 1140.5..1140.5 | 1 | 0.186 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy here, so XGBoost's own device-mismatch fallback), probability

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7652.1 | 7652.1..7652.1 | 1 | - | - | - | 12622.5 | - | accuracy=0.903294, mlogloss=0.281958 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 561543.4 | 561543.4..561543.4 | 1 | 0.014 | - | - | 11547.8 | - | accuracy=0.907768, mlogloss=0.258286 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 76053.9 | 76053.9..76053.9 | 1 | 0.101 | - | - | 12606.4 | - | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 79078.6 | 79078.6..79078.6 | 1 | 0.097 | - | - | 12549.6 | - | accuracy=0.910058, mlogloss=0.245916 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:128000,lightgbm-cpu:76782,ours:128000,xgboost-cpu:88137,xgboost-gpu:88137 spread=0.4001 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | - | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "MultiClass" | - | "MultiClass" | - | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | null | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - | - |
| min_split_gain | - | 0.0 | null | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - | - |
| random_strength | 1.0 | - | 1.0 | - | - |
| reg_alpha | - | 0.0 | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | - | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 | 1.0 |

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 33.4 | 33.4..33.4 | 1 | - | - | - | accuracy=0.903294, accuracy_matches_fit=True, mlogloss=0.281958, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 117.4 | 117.4..117.4 | 1 | 0.284 | - | - | accuracy=0.907768, accuracy_matches_fit=True, mlogloss=0.258286, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 500000 | 560.7 | 560.7..560.7 | 1 | 0.060 | - | - | accuracy=0.910140, accuracy_matches_fit=True, mlogloss=0.246803, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 3682.3 | 3682.3..3682.3 | 1 | 0.009 | - | - | accuracy=0.910058, accuracy_matches_fit=True, mlogloss=0.245916, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 49.3 | 49.3..49.3 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 147.6 | 147.6..147.6 | 1 | 0.334 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 1114.8 | 1114.8..1114.8 | 1 | 0.044 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 7295.8 | 7295.8..7295.8 | 1 | 0.007 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU), the (rows, n_classes) probability matrix

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy), the (rows, n_classes) probability matrix

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), the (rows, n_classes) probability matrix

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (on the CPU whatever device trained it), the (rows, n_classes) probability matrix

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5565.9 | 5565.9..5565.9 | 1 | - | - | - | 7253.4 | - | accuracy=0.596646, mlogloss=1.022664 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 178349.9 | 178349.9..178349.9 | 1 | 0.031 | - | - | 8021.3 | - | accuracy=0.599150, mlogloss=1.012734 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 35861.4 | 35861.4..35861.4 | 1 | 0.155 | - | - | 8463.4 | - | accuracy=0.601128, mlogloss=1.005128 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 41539.9 | 41539.9..41539.9 | 1 | 0.134 | - | - | 8711.3 | - | accuracy=0.601580, mlogloss=1.004282 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:128000,lightgbm-cpu:90172,ours:128000,xgboost-cpu:99745,xgboost-gpu:99745 spread=0.2955 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | - | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "MultiClass" | - | "MultiClass" | - | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | null | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - | - |
| min_split_gain | - | 0.0 | null | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | - | - |
| random_strength | 1.0 | - | 1.0 | - | - |
| reg_alpha | - | 0.0 | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | - | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 | 1.0 |

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 11.6 | 11.6..11.6 | 1 | - | - | - | accuracy=0.596646, accuracy_matches_fit=True, mlogloss=1.022664, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 41.1 | 41.1..41.1 | 1 | 0.282 | - | - | accuracy=0.599150, accuracy_matches_fit=True, mlogloss=1.012734, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 500000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 500000 | 421.6 | 421.6..421.6 | 1 | 0.028 | - | - | accuracy=0.601128, accuracy_matches_fit=True, mlogloss=1.005128, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 3208.1 | 3208.1..3208.1 | 1 | 0.004 | - | - | accuracy=0.601580, accuracy_matches_fit=True, mlogloss=1.004282, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 18.6 | 18.6..18.6 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 41.6 | 41.6..41.6 | 1 | 0.448 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 840.1 | 840.1..840.1 | 1 | 0.022 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 6361.5 | 6361.5..6361.5 | 1 | 0.003 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU), the (rows, n_classes) probability matrix

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy), the (rows, n_classes) probability matrix

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), the (rows, n_classes) probability matrix

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (on the CPU whatever device trained it), the (rows, n_classes) probability matrix

### gbdt-ordered / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-ordered.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 86042.7 | 86042.7..86042.7 | 1 | - | - | - | 11885.8 | - | auc=0.972482, logloss=0.227811 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 273122.4 | 273122.4..273122.4 | 1 | 0.315 | - | - | 11159.1 | - | auc=0.979221, logloss=0.192114 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-ordered arms=catboost-cpu,ours leaves=catboost-cpu:128000,ours:31748 spread=0.7520 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours |
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

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 21.9 | 21.9..21.9 | 1 | - | - | - | auc=0.972482, auc_matches_fit=True, logloss=0.227811, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 44.4 | 44.4..44.4 | 1 | 0.493 | - | - | auc=0.979221, auc_matches_fit=True, logloss=0.192114, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 33.4 | 33.4..33.4 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 118.1 | 118.1..118.1 | 1 | 0.283 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-ordered / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-ordered.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 134646.3 | 134646.3..134646.3 | 1 | - | - | - | 7545.2 | - | auc=0.620358, logloss=0.531519 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 78566.1 | 78566.1..78566.1 | 1 | 1.714 | - | - | 8753.0 | - | auc=0.628918, logloss=0.529083 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-ordered arms=catboost-cpu,ours leaves=catboost-cpu:91326,ours:17836 spread=0.8047 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours |
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

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 3.4 | 3.4..3.4 | 1 | - | - | - | auc=0.620358, auc_matches_fit=True, logloss=0.531519, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 16.5 | 16.5..16.5 | 1 | 0.203 | - | - | auc=0.628918, auc_matches_fit=True, logloss=0.529083, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 5.4 | 5.4..5.4 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 42.3 | 42.3..42.3 | 1 | 0.127 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-pairlogit.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2837.9 | 2837.9..2837.9 | 1 | - | - | - | 13182.5 | - | map=0.844605, ndcg10=0.711712, ndcg5=0.641863 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 31176.3 | 31176.3..31176.3 | 1 | 0.091 | - | - | 12433.4 | - | map=0.846328, ndcg10=0.713361, ndcg5=0.643611 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | NOT-COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 6075.9 | 6075.9..6075.9 | 1 | 0.467 | - | - | 13581.1 | - | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-pairlogit arms=catboost-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:6400,ours:4332,xgboost-cpu:6395,xgboost-gpu:6395 spread=0.3231 verdict=NOT-COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | "No" | - | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" | "depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | - | - |
| leaf_estimation_method | "Newton" | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "PairLogit" | "PairLogit" | - | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 |
| min_child_weight | - | null | 0.0 | 0.0 |
| min_samples_leaf | 1 | 1 | - | - |
| min_split_gain | - | null | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | "Min" | - | - |
| random_strength | 1.0 | 1.0 | - | - |
| reg_alpha | - | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | "Cosine" | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | 1.0 | 1.0 |

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 681250 | 28.4 | 28.4..28.4 | 1 | - | - | - | map=0.844605, map_matches_fit=True, ndcg10=0.711712, ndcg10_matches_fit=True, ndcg5=0.641863, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 681250 | 63.2 | 63.2..63.2 | 1 | 0.449 | - | - | map=0.846328, map_matches_fit=True, ndcg10=0.713361, ndcg10_matches_fit=True, ndcg5=0.643611, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | test | 681250 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 681250 | 47.0 | 47.0..47.0 | 1 | 0.605 | - | - | map=0.872796, map_matches_fit=True, ndcg10=0.738397, ndcg10_matches_fit=True, ndcg5=0.670093, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 36.4 | 36.4..36.4 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 102.9 | 102.9..102.9 | 1 | 0.354 | - | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | NOT-COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 45.2 | 45.2..45.2 | 1 | 0.806 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, catboost-cpu: catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the ranker: a host apply on every box), raw ranking scores

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy), raw ranking scores

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), raw ranking scores

### gbdt-rank-yetirank / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-yetirank.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 18629.8 | 18629.8..18629.8 | 1 | - | - | - | 12084.4 | - | map=0.814902, ndcg10=0.680993, ndcg5=0.615076 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 29452.1 | 29452.1..29452.1 | 1 | 0.633 | - | - | 10688.6 | - | map=0.851990, ndcg10=0.726111, ndcg5=0.660263 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-gpu | xgboost | gpu | opponent | - | - | 0 | - | - | - | - | - | - | yes | COMPARABLE | - | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 6452.1 | 6452.1..6452.1 | 1 | 2.887 | - | - | 11613.2 | - | map=0.842929, ndcg10=0.726256, ndcg5=0.664249 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 5078.0 | 5078.0..5078.0 | 1 | 3.669 | - | - | 11422.7 | - | map=0.858362, ndcg10=0.741515, ndcg5=0.680254 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours, xgboost-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-yetirank arms=catboost-cpu,lightgbm-cpu,ours,xgboost-cpu,xgboost-gpu leaves=catboost-cpu:6400,lightgbm-cpu:6366,ours:6400,xgboost-cpu:6400,xgboost-gpu:6400 spread=0.0053 verdict=COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | xgboost-cpu | xgboost-gpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | xgboost (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "gbtree" | "gbtree" |
| bootstrap_type | "No" | - | "No" | - | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | - | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | - | - |
| leaf_estimation_method | "Newton" | - | "Newton" | - | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "YetiRank" | - | "YetiRank" | - | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 6 | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 | 64 |
| min_child_weight | - | 0.001 | null | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | - | - |
| min_split_gain | - | 0.0 | null | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | - | "Min" | - | - |
| random_strength | 1.0 | - | 1.0 | - | - |
| reg_alpha | - | 0.0 | - | 0.0 | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | - | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | 1.0 | 1.0 |

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-gpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-gpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-gpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-gpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-gpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 681250 | 29.6 | 29.6..29.6 | 1 | - | - | - | map=0.814902, map_matches_fit=True, ndcg10=0.680993, ndcg10_matches_fit=True, ndcg5=0.615076, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| catboost-cpu | test | 681250 | 48.8 | 48.8..48.8 | 1 | 0.607 | - | - | map=0.851990, map_matches_fit=True, ndcg10=0.726111, ndcg10_matches_fit=True, ndcg5=0.660263, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-gpu | test | 681250 | - | - | 0 | - | - | - | - | yes | COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | test | 681250 | 29.9 | 29.9..29.9 | 1 | 0.993 | - | - | map=0.842929, map_matches_fit=True, ndcg10=0.726256, ndcg10_matches_fit=True, ndcg5=0.664249, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cpu | test | 681250 | 139.3 | 139.3..139.3 | 1 | 0.213 | - | - | map=0.858362, map_matches_fit=True, ndcg10=0.741515, ndcg10_matches_fit=True, ndcg5=0.680254, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 36.9 | 36.9..36.9 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 63.3 | 63.3..63.3 | 1 | 0.584 | - | - | - | yes | COMPARABLE | ok |
| xgboost-gpu | large | 1000000 | - | - | 0 | - | - | - | - | yes | COMPARABLE | REFUSED(PyPI xgboost 3.2.0 is a CUDA build; on AMD device=cuda trained on the host CPU; refused by name since ca1f47506) |
| xgboost-cpu | large | 1000000 | 42.8 | 42.8..42.8 | 1 | 0.864 | - | - | - | yes | COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 204.6 | 204.6..204.6 | 1 | 0.181 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, catboost-cpu: catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the ranker: a host apply on every box), raw ranking scores

inference call, xgboost-gpu: xgboost Booster.inplace_predict(host X) on a CUDA booster (no cupy), raw ranking scores

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), raw ranking scores

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (on the CPU whatever device trained it), raw ranking scores

### gbdt-symmetric-1000 / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8894.7 | 8894.7..8894.7 | 1 | - | - | - | 10738.2 | - | auc=0.977075, logloss=0.203751 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 87705.8 | 87705.8..87705.8 | 1 | 0.101 | - | - | 9422.8 | - | auc=0.982309, logloss=0.171620 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-cpu,ours leaves=catboost-cpu:256000,ours:67958 spread=0.7345 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, 1000 trees (CatBoost's own default iteration count; see CATBOOST_DEFAULTS) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours |
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

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 21.5 | 21.5..21.5 | 1 | - | - | - | auc=0.977075, auc_matches_fit=True, logloss=0.203751, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 88.8 | 88.8..88.8 | 1 | 0.242 | - | - | auc=0.982309, auc_matches_fit=True, logloss=0.171620, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 36.5 | 36.5..36.5 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 108.7 | 108.7..108.7 | 1 | 0.336 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-symmetric-1000 / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6114.3 | 6114.3..6114.3 | 1 | - | - | - | 5761.9 | - | auc=0.621310, logloss=0.531329 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 41736.9 | 41736.9..41736.9 | 1 | 0.146 | - | - | 5999.5 | - | auc=0.631642, logloss=0.528267 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-cpu,ours leaves=catboost-cpu:255568,ours:251900 spread=0.0144 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, 1000 trees (CatBoost's own default iteration count; see CATBOOST_DEFAULTS) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours |
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

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 6.8 | 6.8..6.8 | 1 | - | - | - | auc=0.621310, auc_matches_fit=True, logloss=0.531329, logloss_matches_fit=True | yes | COMPARABLE | ok |
| catboost-cpu | test | 500000 | 24.4 | 24.4..24.4 | 1 | 0.277 | - | - | auc=0.631642, auc_matches_fit=True, logloss=0.528267, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 12.5 | 12.5..12.5 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 55.0 | 55.0..55.0 | 1 | 0.228 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5610.5 | 5610.5..5610.5 | 1 | - | - | - | 10739.5 | - | auc=0.977075, logloss=0.203751 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 44398.5 | 44398.5..44398.5 | 1 | 0.126 | - | - | 9400.6 | - | auc=0.979899, logloss=0.188093 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours leaves=catboost-cpu:128000,ours:66958 spread=0.4769 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours |
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

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 20.2 | 20.2..20.2 | 1 | - | - | - | auc=0.977075, auc_matches_fit=True, logloss=0.203751, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 70.1 | 70.1..70.1 | 1 | 0.288 | - | - | auc=0.979899, auc_matches_fit=True, logloss=0.188093, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 39.9 | 39.9..39.9 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 110.8 | 110.8..110.8 | 1 | 0.360 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3215.8 | 3215.8..3215.8 | 1 | - | - | - | 5747.2 | - | auc=0.621310, logloss=0.531329 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 21286.9 | 21286.9..21286.9 | 1 | 0.151 | - | - | 5984.4 | - | auc=0.630269, logloss=0.528650 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, catboost-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours leaves=catboost-cpu:127568,ours:123900 spread=0.0288 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours |
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

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 4.4 | 4.4..4.4 | 1 | - | - | - | auc=0.621310, auc_matches_fit=True, logloss=0.531329, logloss_matches_fit=True | yes | COMPARABLE | ok |
| catboost-cpu | test | 500000 | 16.8 | 16.8..16.8 | 1 | 0.263 | - | - | auc=0.630269, auc_matches_fit=True, logloss=0.528650, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 7.8 | 7.8..7.8 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 42.7 | 42.7..42.7 | 1 | 0.184 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### iforest / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/iforest.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 245.5 | 245.5..245.5 | 1 | - | - | - | 22566.1 | - | auc=0.830358 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 239.9 | 239.9..239.9 | 1 | 1.023 | - | - | 22740.9 | - | auc=0.827914 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-iforest-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:4534 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-iforest-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | false | false |
| contamination | "auto" | "auto" |
| max_depth | null | - |
| max_features | 1.0 | 1.0 |
| max_samples | 256 | 256 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 571.5 | 571.5..571.5 | 1 | - | - | - | auc=0.830358, auc_matches_fit=True | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | test | 500000 | 1997.8 | 1997.8..1997.8 | 1 | 0.286 | - | - | auc=0.827914, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 1010.4 | 1010.4..1010.4 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | large | 1000000 | 3985.8 | 3985.8..3985.8 | 1 | 0.254 | - | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, sklearn-iforest-cpu: sklearn IsolationForest.score_samples(X), n_jobs -1

### iforest / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/iforest.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 173.3 | 173.3..173.3 | 1 | - | - | - | 5653.5 | - | auc=0.551846 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 376.7 | 376.7..376.7 | 1 | 0.460 | - | - | 6800.6 | - | auc=0.552849 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-iforest-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:6131 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-iforest-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | false | false |
| contamination | "auto" | "auto" |
| max_depth | null | - |
| max_features | 1.0 | 1.0 |
| max_samples | 256 | 256 |
| n_estimators | 100 | 100 |
| seed | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 103.7 | 103.7..103.7 | 1 | - | - | - | auc=0.551846, auc_matches_fit=True | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | test | 500000 | 759.3 | 759.3..759.3 | 1 | 0.137 | - | - | auc=0.552849, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 138.1 | 138.1..138.1 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | large | 1000000 | 1475.0 | 1475.0..1475.0 | 1 | 0.094 | - | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, sklearn-iforest-cpu: sklearn IsolationForest.score_samples(X), n_jobs -1

### rf / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/rf.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5106.9 | 5106.9..5106.9 | 1 | - | - | - | 23394.6 | - | auc=0.945385, logloss=0.182017 | yes | COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 90551.1 | 90551.1..90551.1 | 1 | 0.056 | - | - | 26302.6 | - | auc=0.945303, logloss=0.182308 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 21418.1 | 21418.1..21418.1 | 1 | 0.238 | - | - | 25240.8 | - | auc=0.945361, logloss=0.195422 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-rf-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,sklearn-rf-cpu leaves=lightgbm-cpu:125860,ours:125692,sklearn-rf-cpu:125260 spread=0.0048 verdict=COMPARABLE`

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | sklearn-rf-cpu |
|---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - |
| bootstrap | - | true | true |
| class_weight | null | null | null |
| criterion | - | "gini" | "gini" |
| feature_fraction | 1.0 | - | - |
| feature_fraction_bynode | 0.06363636363636363 | - | - |
| learning_rate | 1.0 | - | - |
| max_bin | 128 | 128 | - |
| max_depth | 8 | 8 | 8 |
| max_features | - | "sqrt" | "sqrt" |
| max_leaves | 256 | -1 | null |
| max_samples | - | 1.0 | 1.0 |
| min_child_weight | 0.0 | - | - |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 |
| reg_alpha | 0.0 | - | - |
| reg_lambda | 0.0 | - | - |
| seed | 7 | 7 | 7 |
| subsample | 0.632 | - | - |

accepted difference: sklearn-rf-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 59.8 | 59.8..59.8 | 1 | - | - | - | auc=0.945385, auc_matches_fit=True, logloss=0.182017, logloss_matches_fit=True | yes | COMPARABLE | ok |
| sklearn-rf-cpu | test | 500000 | 637.2 | 637.2..637.2 | 1 | 0.094 | - | - | auc=0.945303, auc_matches_fit=True, logloss=0.182308, logloss_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 700.4 | 700.4..700.4 | 1 | 0.085 | - | - | auc=0.945361, auc_matches_fit=True, logloss=0.195422, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 118.8 | 118.8..118.8 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| sklearn-rf-cpu | large | 1000000 | 1405.6 | 1405.6..1405.6 | 1 | 0.084 | - | - | - | yes | COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1370.8 | 1370.8..1370.8 | 1 | 0.087 | - | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, sklearn-rf-cpu: sklearn RandomForestClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### rf / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/rf.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6754.5 | 6754.5..6754.5 | 1 | - | - | - | 9637.5 | - | auc=0.617838, logloss=0.525953 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 58857.1 | 58857.1..58857.1 | 1 | 0.115 | - | - | 15207.1 | - | auc=0.617678, logloss=0.525532 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 14318.3 | 14318.3..14318.3 | 1 | 0.472 | - | - | 15138.9 | - | auc=0.617040, logloss=0.526421 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU not measured (the probe read the torch allocator for a non-torch arm; fixed at 7c93f50b9)

memory, sklearn-rf-cpu, lightgbm-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,sklearn-rf-cpu leaves=lightgbm-cpu:101602,ours:123231,sklearn-rf-cpu:89688 spread=0.2722 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | sklearn-rf-cpu |
|---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - |
| bootstrap | - | true | true |
| class_weight | null | null | null |
| criterion | - | "gini" | "gini" |
| feature_fraction | 1.0 | - | - |
| feature_fraction_bynode | 0.25 | - | - |
| learning_rate | 1.0 | - | - |
| max_bin | 128 | 128 | - |
| max_depth | 8 | 8 | 8 |
| max_features | - | "sqrt" | "sqrt" |
| max_leaves | 256 | -1 | null |
| max_samples | - | 1.0 | 1.0 |
| min_child_weight | 0.0 | - | - |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 |
| reg_alpha | 0.0 | - | - |
| reg_lambda | 0.0 | - | - |
| seed | 7 | 7 | 7 |
| subsample | 0.632 | - | - |

accepted difference: sklearn-rf-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 12.9 | 12.9..12.9 | 1 | - | - | - | auc=0.617838, auc_matches_fit=True, logloss=0.525953, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| sklearn-rf-cpu | test | 500000 | 302.6 | 302.6..302.6 | 1 | 0.043 | - | - | auc=0.617678, auc_matches_fit=True, logloss=0.525532, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 704.4 | 704.4..704.4 | 1 | 0.018 | - | - | auc=0.617040, auc_matches_fit=True, logloss=0.526421, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 22.6 | 22.6..22.6 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| sklearn-rf-cpu | large | 1000000 | 561.6 | 561.6..561.6 | 1 | 0.040 | - | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1404.5 | 1404.5..1404.5 | 1 | 0.016 | - | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, sklearn-rf-cpu: sklearn RandomForestClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.dbscan.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.14e+06 | 1.14e+06..1.14e+06 | 1 | - | - | - | 5810.5 | 676.5 | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 263657.7 | 263657.7..263657.7 | 1 | 4.325 | - | - | 5971.6 | - | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "rbc" | "brute" |
| eps | 3.0 | 3.0 |
| leaf_size | - | 30 |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| p | - | null |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu algorithm: an exact eps search on every arm: ours 'rbc' (its default), scikit-learn has no 'rbc' and runs 'brute' (the cuML benchmark's cpu_args)

### dbscan / taxi (rows full, shape 1000000x11)

race: failed, driver rc 0, log `logs/classical.dbscan.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('dbscan: the ball-cover neighbourhood has -80883285 edges in one batch, which does not fit the int32 CSR this implementation uses. cuML requires int64 labels for RBC (runner.cuh:1) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | HOST-MEMORY(killed at 143.0 GB: the driver's process tree held 153.6 GB, over 90% of the box's 168.8 GB) (measured this run) |

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "rbc" | "brute" |
| eps | 3.0 | 3.0 |
| leaf_size | - | 30 |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| p | - | null |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu algorithm: an exact eps search on every arm: ours 'rbc' (its default), scikit-learn has no 'rbc' and runs 'brute' (the cuML benchmark's cpu_args)

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| sklearn-cpu | scikit-learn | cpu | opponent | 829354.4 | 829354.4..829354.4 | 1 | - | - | - | 1462.1 | - | n_clusters=52, noise_fraction=0.252570, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | - | "auto" |
| allow_single_cluster | false | false |
| alpha | 1.0 | 1.0 |
| cluster_selection_epsilon | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" |
| leaf_size | - | 40 |
| max_cluster_size | 0 | null |
| metric | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 |
| min_samples | 10 | 10 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu max_cluster_size: no limit on every arm: ours and cuML spell it 0, scikit-learn None

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| sklearn-cpu | scikit-learn | cpu | opponent | 38362.1 | 38362.1..38362.1 | 1 | - | - | - | 334.0 | - | n_clusters=161, noise_fraction=0.134630, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | - | "auto" |
| allow_single_cluster | false | false |
| alpha | 1.0 | 1.0 |
| cluster_selection_epsilon | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" |
| leaf_size | - | 40 |
| max_cluster_size | 0 | null |
| metric | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 |
| min_samples | 10 | 10 |
| seed | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu max_cluster_size: no limit on every arm: ours and cuML spell it 0, scikit-learn None

### kde / istella (rows full, shape 100000x220)

race: done, driver rc 0, log `logs/classical.kde.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 39.6 | 39.6..39.6 | 1 | - | - | - | 2112.1 | 1240.3 | mean_log_likelihood=-222.270586, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 57344.0 | 57344.0..57344.0 | 1 | 0.0006899 | - | - | 466.0 | - | mean_log_likelihood=-226.977407, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" |
| atol | 0.0 | 0.0 |
| bandwidth | 1.0 | 1.0 |
| breadth_first | true | true |
| kernel | "gaussian" | "gaussian" |
| leaf_size | 40 | 40 |
| metric | "euclidean" | "euclidean" |
| rtol | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" |

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc 0, log `logs/classical.kde.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 29.2 | 29.2..29.2 | 1 | - | - | - | 2030.5 | 1160.3 | mean_log_likelihood=-14.826460, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7083.5 | 7083.5..7083.5 | 1 | 0.004 | - | - | 209.0 | - | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" |
| atol | 0.0 | 0.0 |
| bandwidth | 1.0 | 1.0 |
| breadth_first | true | true |
| kernel | "gaussian" | "gaussian" |
| leaf_size | 40 | 40 |
| metric | "euclidean" | "euclidean" |
| rtol | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" |

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 741.4 | 741.4..741.4 | 1 | - | - | - | 4186.5 | 676.5 | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5614.6 | 5614.6..5614.6 | 1 | 0.132 | - | - | 5794.7 | - | inertia=6.049e+17, inertia_over_ours=0.999759, n_iter=24 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2056.2 | 2056.2..2056.2 | 1 | 0.361 | - | - | 5131.2 | 3460.8 | inertia=5.991e+17, inertia_over_ours=0.990156, n_iter=55 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | - | "lloyd" | - |
| init | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 300 | 300 | 300 |
| metric | "euclidean" | - | "euclidean" |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| oversampling_factor | 0.0 | - | - |
| seed | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 41.2 | 41.2..41.2 | 1 | - | - | - | eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 23.3 | 23.3..23.3 | 1 | 1.764 | - | - | agreement_vs_ours=0.031282, bits_equal_vs_ours=False, eval_inertia=1.418e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.8 | 0.8..0.8 | 1 | 53.873 | - | - | agreement_vs_ours=0.038610, bits_equal_vs_ours=False, eval_inertia=1.402e+17, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 420.0 | 420.0..420.0 | 1 | - | - | - | 2262.9 | 676.5 | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2297.6 | 2297.6..2297.6 | 1 | 0.183 | - | - | 772.1 | - | inertia=3.093e+08, inertia_over_ours=0.999937, n_iter=58 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 850.7 | 850.7..850.7 | 1 | 0.494 | - | - | 3193.5 | 459.9 | inertia=3.129e+08, inertia_over_ours=1.011535, n_iter=81 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | - | "lloyd" | - |
| init | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 300 | 300 | 300 |
| metric | "euclidean" | - | "euclidean" |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| oversampling_factor | 0.0 | - | - |
| seed | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 4.5 | 4.5..4.5 | 1 | - | - | - | eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 4.6 | 4.6..4.6 | 1 | 0.974 | - | - | agreement_vs_ours=0.003656, bits_equal_vs_ours=False, eval_inertia=4.816e+07, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | 9.584 | - | - | agreement_vs_ours=0.496926, bits_equal_vs_ours=False, eval_inertia=4.926e+07, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### knn / istella (rows full, shape 400000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1183.8 | 1183.8..1183.8 | 1 | - | - | - | 2369.0 | 1594.4 | recall_at_k=0.976250, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 536.9 | 536.9..536.9 | 1 | 2.205 | - | - | 587.8 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 31.9 | 31.9..31.9 | 1 | 37.056 | - | - | 3187.5 | 3884.6 | recall_at_k=0.978680, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | 30 | - |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 | 64 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1282.6 | 1282.6..1282.6 | 1 | - | - | - | 2046.7 | 1276.4 | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 381.1 | 381.1..381.1 | 1 | 3.366 | - | - | 238.1 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 26.9 | 26.9..26.9 | 1 | 47.760 | - | - | 2865.2 | 3242.7 | recall_at_k=0.999730, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | 30 | - |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 | 64 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.ols.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1319.9 | 1319.9..1319.9 | 1 | - | - | - | 5892.6 | 674.5 | finite=True, r2=0.331944, rmse=0.682027 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3980.6 | 3980.6..3980.6 | 1 | 0.332 | - | - | 5793.0 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2338.0 | 2338.0..2338.0 | 1 | 0.565 | - | - | 5634.6 | 5303.6 | finite=False, r2=nan, rmse=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 41.8 | 41.8..41.8 | 1 | 31.561 | - | - | 5038.4 | 3649.4 | finite=True, r2=0.151604, rmse=0.768589 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu | torch-gpu-eigh |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) | torch (declared) |
| fit_intercept | true | true | true | true |
| positive | - | false | - | - |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |
| tol | - | 1e-06 | - | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 12.3 | 12.3..12.3 | 1 | - | - | - | predict_max_rel_err_own_fp64=9.581e-07, r2_eval=0.331944, rmse_eval=0.682027 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 18.5 | 18.5..18.5 | 1 | 0.665 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=7.010309, predict_max_rel_err_own_fp64=7.648e-08, r2_eval=0.001881, rmse_eval=0.833655 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | 23.721 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=nan, predict_max_rel_err_own_fp64=nan, r2_eval=nan, rmse_eval=nan | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu-eigh | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | 25.155 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=6.699501, predict_max_rel_err_own_fp64=4.831e-07, r2_eval=0.151604, rmse_eval=0.768589 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 156.9 | 156.9..156.9 | 1 | - | - | - | 2414.8 | 674.4 | finite=True, r2=0.908837, rmse=4.696466 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 304.7 | 304.7..304.7 | 1 | 0.515 | - | - | 782.5 | - | finite=True, r2=0.724850, rmse=8.159187 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 109.7 | 109.7..109.7 | 1 | 1.430 | - | - | 3535.6 | 695.3 | finite=True, r2=0.908840, rmse=4.696376 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |
| torch-gpu-eigh | torch | gpu | opponent | 33.8 | 33.8..33.8 | 1 | 4.638 | - | - | 3017.1 | 572.0 | finite=True, r2=0.908822, rmse=4.696849 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu, torch-gpu-eigh: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu | torch-gpu-eigh |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) | torch (declared) |
| fit_intercept | true | true | true | true |
| positive | - | false | - | - |
| seed | "none (deterministic)" | "none (deterministic)" | 7 | 7 |
| tol | - | 1e-06 | - | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 1.6 | 1.6..1.6 | 1 | - | - | - | predict_max_rel_err_own_fp64=8.387e-08, r2_eval=0.908837, rmse_eval=4.696466 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 1.8 | 1.8..1.8 | 1 | 0.898 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=197.960266, predict_max_rel_err_own_fp64=1.385e-07, r2_eval=0.724850, rmse_eval=8.159187 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.5 | 0.5..0.5 | 1 | 3.448 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.063354, predict_max_rel_err_own_fp64=5.913e-08, r2_eval=0.908840, rmse_eval=4.696376 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |
| torch-gpu-eigh | Xq | 500000 | 0.4 | 0.4..0.4 | 1 | 3.706 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.555836, predict_max_rel_err_own_fp64=8.007e-08, r2_eval=0.908822, rmse_eval=4.696849 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

inference call, torch-gpu-eigh: torch Xq @ coef + intercept; Xq uploaded before the clock, which ends at the device synchronize

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 289.2 | 289.2..289.2 | 1 | - | - | - | 4171.7 | 674.4 | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 621.7 | 621.7..621.7 | 1 | 0.465 | - | - | 2342.9 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 20.3 | 20.3..20.3 | 1 | 14.243 | - | - | 5043.3 | 3505.8 | explained_variance_ratio_sum=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| n_components | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | 0.0 | - |
| whiten | false | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 17.7 | 17.7..17.7 | 1 | - | - | - | transform_max_rel_err_own_fp64=3.75e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 24.7 | 24.7..24.7 | 1 | 0.715 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=64529.218750, transform_max_rel_err_own_fp64=3.857e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.6 | 0.6..0.6 | 1 | 29.422 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=1.678e+06, transform_max_rel_err_own_fp64=3.575e-07 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20.0 | 20.0..20.0 | 1 | - | - | - | 2232.8 | 674.3 | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 82.2 | 82.2..82.2 | 1 | 0.243 | - | - | 401.6 | - | explained_variance_ratio_sum=0.999995 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 10.9 | 10.9..10.9 | 1 | 1.833 | - | - | 3008.7 | 412.0 | explained_variance_ratio_sum=0.999997 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-gpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| n_components | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | 0.0 | - |
| whiten | false | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 6.9 | 6.9..6.9 | 1 | - | - | - | transform_max_rel_err_own_fp64=1.085e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 5.6 | 5.6..5.6 | 1 | 1.220 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.897790, transform_max_rel_err_own_fp64=1.299e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 0.3 | 0.3..0.3 | 1 | 21.310 | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=322.178085, transform_max_rel_err_own_fp64=9.597e-08 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: torch (Xq - mean) @ components.T; Xq uploaded before the clock, which ends at the device synchronize

### svc / istella (rows full, shape 10000x220)

race: done, driver rc 0, log `logs/classical.svc.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 65.6 | 65.6..65.6 | 1 | - | - | - | 2069.9 | 3246.7 | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 854.5 | 854.5..854.5 | 1 | 0.077 | - | - | 335.4 | - | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| seed | null | 7 |
| tol | 0.001 | 0.001 |

accepted difference: ours seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 10000 | 7.3 | 7.3..7.3 | 1 | - | - | - | accuracy_eval=0.922200 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 10000 | 1940.9 | 1940.9..1940.9 | 1 | 0.004 | - | - | accuracy_eval=0.922200, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn SVC.predict(Xq), host rows in, host result out

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc 0, log `logs/classical.svc.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1454.7 | 1454.7..1454.7 | 1 | - | - | - | 2030.5 | 3246.8 | accuracy=0.767500, n_support=5527 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1727.5 | 1727.5..1727.5 | 1 | 0.842 | - | - | 455.3 | - | accuracy=0.767500, n_support=5675 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| coef0 | 0.0 | 0.0 |
| degree | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| max_iter | -1 | -1 |
| seed | null | 7 |
| tol | 0.001 | 0.001 |

accepted difference: ours seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 10000 | 4.0 | 4.0..4.0 | 1 | - | - | - | accuracy_eval=0.767500 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 10000 | 1005.9 | 1005.9..1005.9 | 1 | 0.004 | - | - | accuracy_eval=0.767500, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn SVC.predict(Xq), host rows in, host result out

## Classical, wave 2

### agglomerative / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.agglomerative.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 112.3 | 112.3..112.3 | 1 | - | - | - | 2961.0 | 676.5 | n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3592.6 | 3592.6..3592.6 | 1 | 0.031 | - | - | 1146.4 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

config: cuML benchmark (RAPIDS), AgglomerativeClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (attributes) | sklearn (get_params) |
| linkage | "single" | "single" |
| metric | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| seed | "none (deterministic)" | "none (deterministic)" |

### agglomerative / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.agglomerative.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15.4 | 15.4..15.4 | 1 | - | - | - | 2076.1 | 676.5 | n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 241.2 | 241.2..241.2 | 1 | 0.064 | - | - | 245.3 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

config: cuML benchmark (RAPIDS), AgglomerativeClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (attributes) | sklearn (get_params) |
| linkage | "single" | "single" |
| metric | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 |
| seed | "none (deterministic)" | "none (deterministic)" |

### arima / synthetic (rows full, shape Yfit 64x2000; Yhold 64x100)

race: done, driver rc 0, log `logs/classical2.arima.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1334.6 | 1334.6..1334.6 | 1 | - | - | - | 2025.2 | 1608.5 | forecast_rmse=1.515518, insample_rmse=0.999342, mean_aic=5680.976967, mean_llf=-2836.488483 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 184.2 | 184.2..184.2 | 1 | 7.244 | - | - | 194.2 | - | forecast_rmse=1.515423, insample_rmse=0.999338, mean_aic=5680.957160, mean_llf=-2836.478580 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: order=(1,0,1), seasonal_order=(0,0,0,0), trend='c' (cuML fit_intercept=True), maxiter=1000, maximum likelihood. Rows: 64 synthetic ARMA(1,1) series, 2000 fit points, 100 held out. Timed: fit of every series.

mismatch: ours and cuML fit the whole batch in one call; statsmodels fits one series per call (the state-space model, L-BFGS), spread over every core with joblib

mismatch: statsmodels enforce_stationarity and enforce_invertibility at its default (True); ours and cuML have no such parameter

mismatch: seed: no arm has a seed argument (maximum likelihood)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | statsmodels-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | statsmodels (declared) |
| max_iter | 1000 | 1000 |
| order | [1, 0, 1] | [1, 0, 1] |
| seasonal_order | [0, 0, 0, 0] | [0, 0, 0, 0] |
| seed | "none (deterministic)" | "none (deterministic)" |
| trend | "c" | "c" |

### elasticnet / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1769.8 | 1769.8..1769.8 | 1 | - | - | - | 3791.1 | 674.4 | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2177.9 | 2177.9..2177.9 | 1 | 0.813 | - | - | 2803.2 | - | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.1 | 0.1 |
| fit_intercept | true | true |
| l1_ratio | 0.5 | 0.5 |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | false | false |
| seed | null | 7 |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | - |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

### elasticnet / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 58.6 | 58.6..58.6 | 1 | - | - | - | 2116.7 | 674.4 | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 113.1 | 113.1..113.1 | 1 | 0.519 | - | - | 332.3 | - | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.1 | 0.1 |
| fit_intercept | true | true |
| l1_ratio | 0.5 | 0.5 |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | false | false |
| seed | null | 7 |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | - |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

### ets / synthetic (rows full, shape Yfit 64x1440; Yhold 64x48)

race: done, driver rc 0, log `logs/classical2.ets.synthetic.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 337.8 | 337.8..337.8 | 1 | - | - | - | 2027.8 | 678.4 | forecast_rmse=0.984392, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 631.5 | 631.5..631.5 | 1 | 0.535 | - | - | 193.1 | - | forecast_rmse=0.984664, insample_rmse=0.992056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, statsmodels-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: trend additive, seasonal additive, seasonal_periods=24, initialization_method='estimated'; ours and cuML start_periods=2, eps=2.24e-3; statsmodels damped_trend=False, use_boxcox=False. Rows: 64 synthetic hourly series, period 24, 1440 fit points, 48 held out. Timed: construct + fit of every series.

mismatch: initialization: ours 'estimated' (its default, statsmodels' definition), statsmodels 'estimated'; cuML has only its heuristic start (start_periods=2), so its row fits the older initialization

mismatch: cuML returns no in-sample predictions; that quality cell is empty

mismatch: trend: ours and cuML are additive-trend with no parameter; statsmodels trend='additive'. eps is ours' and cuML's only; statsmodels fit() uses its own optimizer

mismatch: seed: no arm has a seed argument

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | statsmodels-cpu |
|---|---||---|---|
| library (source) | mojolearn (attributes) | statsmodels (declared) |
| damped_trend | - | false |
| eps | 0.00224 | - |
| initialization_method | "estimated" | "estimated" |
| seasonal | "additive" | "additive" |
| seasonal_periods | 24 | 24 |
| seed | "none (deterministic)" | "none (deterministic)" |
| start_periods | 2 | - |
| trend | - | "additive" |

### gmm / istella (rows full, shape X 100000x200; Xq 20000x200; _dropped_constant_columns 20)

race: done, driver rc 0, log `logs/classical2.gmm.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1691.4 | 1691.4..1691.4 | 1 | - | - | - | 3133.2 | 3249.0 | bic=-3.851e+07, mean_log_likelihood=200.794403, n_iter=24 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 19489.5 | 19489.5..19489.5 | 1 | 0.087 | - | - | 1544.3 | - | bic=-3.901e+07, mean_log_likelihood=200.768331, n_iter=39 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 302.6 | 302.6..302.6 | 1 | - | - | - | 2097.3 | 678.8 | bic=-3.67e+06, mean_log_likelihood=12.861940, n_iter=32 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(scikit-learn 1.7.2 / OpenBLAS collapses a component at reg_covar 1e-6 on this box; it fits on the Macs; the library said: error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.gpc.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8587.6 | 8587.6..8587.6 | 1 | - | - | - | 3124.8 | 3244.6 | accuracy=0.901333, logloss=0.232590, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2549.7 | 2549.7..2549.7 | 1 | 3.368 | - | - | 1643.0 | - | accuracy=0.901333, logloss=0.232597, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.gpc.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6442.0 | 6442.0..6442.0 | 1 | - | - | - | 2239.4 | 3244.6 | accuracy=0.761000, logloss=0.541286, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1653.6 | 1653.6..1653.6 | 1 | 3.896 | - | - | 755.5 | - | accuracy=0.761000, logloss=0.541358, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.gpr.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1138.0 | 1138.0..1138.0 | 1 | - | - | - | 3348.7 | 3424.0 | finite=True, mean_log_predictive_density=-9.285754, r2=0.235346, rmse=0.760439 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 443.7 | 443.7..443.7 | 1 | 2.565 | - | - | 1430.9 | - | finite=True, mean_log_predictive_density=-9.287148, r2=0.235368, rmse=0.760428 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.gpr.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1175.0 | 1175.0..1175.0 | 1 | - | - | - | 2467.3 | 3424.0 | finite=True, mean_log_predictive_density=-311.458394, r2=0.889630, rmse=5.041639 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 162.8 | 162.8..162.8 | 1 | 7.219 | - | - | 540.4 | - | finite=True, mean_log_predictive_density=-311.539594, r2=0.889629, rmse=5.041653 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.ivf.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 264830.8 | 264830.8..264830.8 | 1 | - | - | - | 4008.0 | 678.6 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 3704.2 | 3704.2..3704.2 | 1 | 71.494 | - | - | 1286.6 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours |
|---|---||---|---|
| library (source) | faiss (declared) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 |
| nlist | 1024 | 1024 |
| nprobe | 32 | 32 |
| seed | 7 | 7 |

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/classical2.ivf.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20382.2 | 20382.2..20382.2 | 1 | - | - | - | 2141.7 | 678.6 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 143.0 | 143.0..143.0 | 1 | 142.581 | - | - | 139.0 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, faiss-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours |
|---|---||---|---|
| library (source) | faiss (declared) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 |
| nlist | 1024 | 1024 |
| nprobe | 32 | 32 |
| seed | 7 | 7 |

### kernel-ridge / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11795.2 | 11795.2..11795.2 | 1 | - | - | - | 2986.2 | 3244.5 | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 843.7 | 843.7..843.7 | 1 | 13.981 | - | - | 2336.5 | - | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" |
| seed | "none (deterministic)" | "none (deterministic)" |

### kernel-ridge / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12069.8 | 12069.8..12069.8 | 1 | - | - | - | 2076.9 | 3246.5 | finite=True, r2=0.726543, rmse=8.330373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 825.3 | 825.3..825.3 | 1 | 14.625 | - | - | 1449.1 | - | finite=True, r2=0.726543, rmse=8.330374 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 |
| degree | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" |
| seed | "none (deterministic)" | "none (deterministic)" |

### knn-clf / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 44.6 | 44.6..44.6 | 1 | - | - | - | 3126.5 | 856.4 | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 281.5 | 281.5..281.5 | 1 | 0.158 | - | - | 1335.5 | - | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" |
| leaf_size | - | 30 |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### knn-clf / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26.0 | 26.0..26.0 | 1 | - | - | - | 2089.3 | 676.4 | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 102.7 | 102.7..102.7 | 1 | 0.253 | - | - | 267.3 | - | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" |
| leaf_size | - | 30 |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### knn-reg / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 34.1 | 34.1..34.1 | 1 | - | - | - | 3123.8 | 856.4 | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 284.2 | 284.2..284.2 | 1 | 0.120 | - | - | 1333.6 | - | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" |
| leaf_size | - | 30 |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### knn-reg / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.8 | 13.8..13.8 | 1 | - | - | - | 2084.1 | 676.3 | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 95.2 | 95.2..95.2 | 1 | 0.145 | - | - | 265.5 | - | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" |
| leaf_size | - | 30 |
| metric | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 |
| p | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" |

### lasso / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4636.7 | 4636.7..4636.7 | 1 | - | - | - | 3791.1 | 674.4 | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7075.0 | 7075.0..7075.0 | 1 | 0.655 | - | - | 2803.7 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 |
| fit_intercept | true | true |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | false | false |
| seed | null | 7 |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | - |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

### lasso / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 55.5 | 55.5..55.5 | 1 | - | - | - | 2116.4 | 674.4 | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 123.8 | 123.8..123.8 | 1 | 0.449 | - | - | 332.5 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 |
| fit_intercept | true | true |
| max_iter | 1000 | 1000 |
| positive | false | false |
| precompute | false | false |
| seed | null | 7 |
| selection | "cyclic" | "cyclic" |
| solver | "cd" | - |
| tol | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

### linearsvc / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1977.3 | 1977.3..1977.3 | 1 | - | - | - | 3019.3 | 674.4 | accuracy=0.923480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 802255.8 | 802255.8..802255.8 | 1 | 0.002 | - | - | 6004.6 | - | accuracy=0.923540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| loss | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | - |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 |

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

### linearsvc / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 119.3 | 119.3..119.3 | 1 | - | - | - | 2142.3 | 674.4 | accuracy=0.763330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 890.7 | 890.7..890.7 | 1 | 0.134 | - | - | 634.0 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| loss | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | - |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 |

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

### linearsvr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2789.8 | 2789.8..2789.8 | 1 | - | - | - | 2952.2 | 674.4 | finite=True, r2=-0.106754, rmse=0.878792 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 792181.6 | 792181.6..792181.6 | 1 | 0.004 | - | - | 6008.3 | - | finite=True, r2=-0.025729, rmse=0.846012 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 |
| fit_intercept | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | - |
| penalty | "l2" | - |
| seed | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 |

### linearsvr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 306.0 | 306.0..306.0 | 1 | - | - | - | 2075.0 | 674.4 | finite=True, r2=0.899813, rmse=5.041302 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 106987.6 | 106987.6..106987.6 | 1 | 0.003 | - | - | 637.4 | - | finite=True, r2=0.899803, rmse=5.041552 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 |
| fit_intercept | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 |
| penalized_intercept | false | - |
| penalty | "l2" | - |
| seed | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 |

### logreg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11491.7 | 11491.7..11491.7 | 1 | - | - | - | 2958.9 | 674.4 | accuracy=0.924540, logloss=0.181245, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20612.3 | 20612.3..20612.3 | 1 | 0.558 | - | - | 2833.9 | - | accuracy=0.924590, logloss=0.181264, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| l1_ratio | null | null |
| max_iter | 1000 | 1000 |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | 7 |
| solver | "qn" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu solver: L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'

### logreg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 414.3 | 414.3..414.3 | 1 | - | - | - | 2082.1 | 674.4 | accuracy=0.763340, logloss=0.538984, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 233.9 | 233.9..233.9 | 1 | 1.771 | - | - | 367.1 | - | accuracy=0.763320, logloss=0.538980, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 |
| class_weight | null | null |
| fit_intercept | true | true |
| l1_ratio | null | null |
| max_iter | 1000 | 1000 |
| penalty | "l2" | "l2" |
| seed | "none (deterministic)" | 7 |
| solver | "qn" | "lbfgs" |
| tol | 0.0001 | 0.0001 |

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu solver: L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'

### nystroem / istella (rows full, shape X 100000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.nystroem.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 574.0 | 574.0..574.0 | 1 | - | - | - | 1792.2 | - | kernel_rel_error=0.038958 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.nystroem.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 376.5 | 376.5..376.5 | 1 | - | - | - | 794.2 | - | kernel_rel_error=0.044369 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.rbf-sampler.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 69.9 | 69.9..69.9 | 1 | - | - | - | 3337.3 | 3248.4 | kernel_rel_error=0.141980 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 25.5 | 25.5..25.5 | 1 | 2.736 | - | - | 1490.2 | - | kernel_rel_error=0.137405 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.rbf-sampler.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 36.4 | 36.4..36.4 | 1 | - | - | - | 2278.6 | 3246.3 | kernel_rel_error=0.108549 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 16.4 | 16.4..16.4 | 1 | 2.216 | - | - | 449.0 | - | kernel_rel_error=0.083775 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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

race: done, driver rc 0, log `logs/classical2.ridge.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1157.5 | 1157.5..1157.5 | 1 | - | - | - | 3794.1 | 674.5 | finite=True, r2=0.328682, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4288.8 | 4288.8..4288.8 | 1 | 0.270 | - | - | 8688.9 | - | finite=True, r2=0.328676, rmse=0.684426 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 |
| fit_intercept | true | true |
| max_iter | - | null |
| normalize | false | - |
| positive | - | false |
| seed | "none (deterministic)" | 7 |
| solver | "eig" | "cholesky" |
| tol | - | 0.0001 |

accepted difference: sklearn-cpu solver: ours and cuML 'eig' (eigendecomposition of the normal equations); scikit-learn has no 'eig' and runs 'cholesky' on the same normal equations

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.ridge.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.8 | 54.8..54.8 | 1 | - | - | - | 2120.2 | 674.5 | finite=True, r2=0.908983, rmse=4.805042 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 33.2 | 33.2..33.2 | 1 | 1.650 | - | - | 291.3 | - | finite=True, r2=0.908983, rmse=4.805056 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 |
| fit_intercept | true | true |
| max_iter | - | null |
| normalize | false | - |
| positive | - | false |
| seed | "none (deterministic)" | 7 |
| solver | "eig" | "cholesky" |
| tol | - | 0.0001 |

accepted difference: sklearn-cpu solver: ours and cuML 'eig' (eigendecomposition of the normal equations); scikit-learn has no 'eig' and runs 'cholesky' on the same normal equations

### spectral-embedding / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 159.9 | 159.9..159.9 | 1 | - | - | - | 2076.5 | 858.7 | trustworthiness_k15=0.799386 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8794.3 | 8794.3..8794.3 | 1 | 0.018 | - | - | 460.1 | - | trustworthiness_k15=0.812682 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows: the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" |
| gamma | null | null |
| n_components | 2 | 2 |
| n_neighbors | 10 | 10 |
| seed | 7 | 7 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral-embedding / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 90.6 | 90.6..90.6 | 1 | - | - | - | 2043.0 | 678.6 | trustworthiness_k15=0.884889 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2875.9 | 2875.9..2875.9 | 1 | 0.032 | - | - | 255.2 | - | trustworthiness_k15=0.898012 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows: the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" |
| gamma | null | null |
| n_components | 2 | 2 |
| n_neighbors | 10 | 10 |
| seed | 7 | 7 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.spectral.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 115.1 | 115.1..115.1 | 1 | - | - | - | 2982.3 | 858.9 | n_clusters=8, silhouette=0.147668 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1577.8 | 1577.8..1577.8 | 1 | 0.073 | - | - | 1229.8 | - | ari_vs_ours=0.999826, n_clusters=8, silhouette=0.147699 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, random_state=42 (the cuML benchmark's SpectralClustering), assign_labels='kmeans', n_components=8. Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

config: cuML benchmark (RAPIDS), SpectralClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" |
| assign_labels | "kmeans" | "kmeans" |
| coef0 | - | 1 |
| degree | - | 3 |
| gamma | null | 1.0 |
| n_clusters | 8 | 8 |
| n_components | 8 | 8 |
| n_init | 1 | 1 |
| n_neighbors | 10 | 10 |
| seed | 42 | 42 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.spectral.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 129.4 | 129.4..129.4 | 1 | - | - | - | 2087.0 | 678.8 | n_clusters=8, silhouette=0.039910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 932.7 | 932.7..932.7 | 1 | 0.139 | - | - | 283.3 | - | ari_vs_ours=0.582313, n_clusters=8, silhouette=0.089894 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, random_state=42 (the cuML benchmark's SpectralClustering), assign_labels='kmeans', n_components=8. Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

config: cuML benchmark (RAPIDS), SpectralClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" |
| assign_labels | "kmeans" | "kmeans" |
| coef0 | - | 1 |
| degree | - | 3 |
| gamma | null | 1.0 |
| n_clusters | 8 | 8 |
| n_components | 8 | 8 |
| n_init | 1 | 1 |
| n_neighbors | 10 | 10 |
| seed | 42 | 42 |

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### svr / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.svr.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 128.9 | 128.9..128.9 | 1 | - | - | - | 3001.5 | 3246.7 | finite=True, r2=0.318258, rmse=0.680816 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1073.0 | 1073.0..1073.0 | 1 | 0.120 | - | - | 1289.4 | - | finite=True, r2=0.318248, rmse=0.680821 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
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

race: done, driver rc 0, log `logs/classical2.svr.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 113.4 | 113.4..113.4 | 1 | - | - | - | 2079.5 | 3246.7 | finite=True, r2=0.767551, rmse=7.680395 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1296.9 | 1296.9..1296.9 | 1 | 0.087 | - | - | 584.8 | - | finite=True, r2=0.767550, rmse=7.680409 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
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

race: done, driver rc 0, log `logs/classical2.tsvd.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 66.4 | 66.4..66.4 | 1 | - | - | - | 2864.6 | 674.4 | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.002554 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 689.8 | 689.8..689.8 | 1 | 0.096 | - | - | 2422.2 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.000122 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "covariance_eigh" | "arpack" |
| n_components | 10 | 10 |
| n_iter | 5 | 5 |
| seed | 7 | 7 |
| tol | 0.0 | 0.0 |

accepted difference: sklearn-cpu algorithm: scikit-learn TruncatedSVD has no 'covariance_eigh'; it runs 'arpack' at tol=0

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc 0, log `logs/classical2.tsvd.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.2 | 14.2..14.2 | 1 | - | - | - | 2066.2 | 674.4 | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 205.6 | 205.6..205.6 | 1 | 0.069 | - | - | 460.6 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "covariance_eigh" | "arpack" |
| n_components | 10 | 10 |
| n_iter | 5 | 5 |
| seed | 7 | 7 |
| tol | 0.0 | 0.0 |

accepted difference: sklearn-cpu algorithm: scikit-learn TruncatedSVD has no 'covariance_eigh'; it runs 'arpack' at tol=0

### umap / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.umap.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 498.5 | 498.5..498.5 | 1 | - | - | - | 2086.7 | 858.7 | trustworthiness_k15=0.979906 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 7336.5 | 7336.5..7336.5 | 1 | 0.068 | - | - | 670.6 | - | trustworthiness_k15=0.977439 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1326.8 | 1326.8..1326.8 | 1 | 0.376 | - | - | 701.2 | - | trustworthiness_k15=0.977474 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows: 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

config: cuML benchmark (RAPIDS), UMAP-Unsupervised (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | umap-learn-cpu | umap-learn-cpu-unseeded |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | umap-learn (get_params) | umap-learn (get_params) |
| init | "spectral" | "spectral" | "spectral" |
| learning_rate | 1.0 | 1.0 | 1.0 |
| local_connectivity | 1.0 | 1.0 | 1.0 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_dist | 0.1 | 0.1 | 0.1 |
| n_components | 2 | 2 | 2 |
| n_epochs | 500 | 500 | 500 |
| n_neighbors | 5 | 5 | 5 |
| negative_sample_rate | 5 | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 | 1.0 |
| seed | 7 | 7 | null |
| set_op_mix_ratio | 1.0 | 1.0 | 1.0 |
| spread | 1.0 | 1.0 | 1.0 |

accepted difference: umap-learn-cpu-unseeded seed: raced unseeded on purpose: seeded umap-learn runs one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)

### umap / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.umap.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 188.1 | 188.1..188.1 | 1 | - | - | - | 2040.1 | 678.6 | trustworthiness_k15=0.990480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 6542.2 | 6542.2..6542.2 | 1 | 0.029 | - | - | 648.9 | - | trustworthiness_k15=0.990061 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1115.6 | 1115.6..1115.6 | 1 | 0.169 | - | - | 661.8 | - | trustworthiness_k15=0.990396 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows: 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

config: cuML benchmark (RAPIDS), UMAP-Unsupervised (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | umap-learn-cpu | umap-learn-cpu-unseeded |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | umap-learn (get_params) | umap-learn (get_params) |
| init | "spectral" | "spectral" | "spectral" |
| learning_rate | 1.0 | 1.0 | 1.0 |
| local_connectivity | 1.0 | 1.0 | 1.0 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_dist | 0.1 | 0.1 | 0.1 |
| n_components | 2 | 2 | 2 |
| n_epochs | 500 | 500 | 500 |
| n_neighbors | 5 | 5 | 5 |
| negative_sample_rate | 5 | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 | 1.0 |
| seed | 7 | 7 | null |
| set_op_mix_ratio | 1.0 | 1.0 | 1.0 |
| spread | 1.0 | 1.0 | 1.0 |

accepted difference: umap-learn-cpu-unseeded seed: raced unseeded on purpose: seeded umap-learn runs one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)

## Neural

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 27.3 | 27.3..27.3 | 1 | - | - | - | 968.1 | 3244.3 | max_rel_err_vs_fp64=2.399e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 13.5 | 13.5..13.5 | 1 | 2.017 | - | - | 2895.2 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 13.7 | 13.7..13.7 | 1 | 1.989 | - | - | 3033.7 | 268.0 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.802e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 11.2 | 11.2..11.2 | 1 | 2.444 | - | - | 4284.9 | 300.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 12.8 | 12.8..12.8 | 1 | 2.123 | - | - | 4500.5 | 300.0 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-forward.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 89.7 | 89.7..89.7 | 1 | - | - | - | 2351.7 | 7284.5 | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-compile-fp32 | torch | gpu | opponent | 14.1 | 14.1..14.1 | 1 | 6.377 | - | - | 3201.9 | 222.5 | max_abs_diff_vs_ours=1.311e-06, max_rel_diff_vs_ours=1.8e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 30.3 | 30.3..30.3 | 1 | 2.958 | - | - | 4433.6 | 194.5 | max_abs_diff_vs_ours=0.006870, max_rel_diff_vs_ours=0.009432, mean_nll=9.018664 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 16.4 | 16.4..16.4 | 1 | 5.459 | - | - | 4312.2 | 240.5 | max_abs_diff_vs_ours=0.006042, max_rel_diff_vs_ours=0.008295, mean_nll=9.018647 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:28:57Z on mojolearn-steward-do-amd, gpu (AMD Instinct Mi325X VF))) |
| torch-eager-fp32 | torch | gpu | opponent | 19.6 | 19.6..19.6 | 1 | 4.588 | - | - | 3104.5 | 241.0 | max_abs_diff_vs_ours=1.311e-06, max_rel_diff_vs_ours=1.8e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:28:57Z on mojolearn-steward-do-amd, gpu (AMD Instinct Mi325X VF))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-compile-fp32, torch-compile-bf16, torch-eager-bf16, torch-eager-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-train-step.bytes.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 123.7 | 123.7..123.7 | 1 | - | - | - | 2124.2 | 6505.1 | loss_first_step=9.018733, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 18.1 | 18.1..18.1 | 1 | 6.833 | - | - | 3259.3 | 931.1 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018732, loss_last_abs_diff_vs_ours=0.000000, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 14.6 | 14.6..14.6 | 1 | 8.469 | - | - | 3310.6 | 780.1 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=9.018733, loss_last_abs_diff_vs_ours=0.000000, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 13.2 | 13.2..13.2 | 1 | 9.396 | - | - | 4834.4 | 774.1 | loss_first_abs_diff_vs_ours=8.678e-05, loss_first_step=9.018646, loss_last_abs_diff_vs_ours=0.002108, loss_last_step=8.420303, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 39.6 | 39.6..39.6 | 1 | 3.125 | - | - | 4894.5 | 602.6 | loss_first_abs_diff_vs_ours=6.962e-05, loss_first_step=9.018663, loss_last_abs_diff_vs_ours=0.002758, loss_last_step=8.419653, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### mamba1-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 18.8 | 18.8..18.8 | 1 | - | - | - | 727.3 | 3246.6 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 100.9 | 100.9..100.9 | 1 | 0.186 | - | - | 3392.1 | 474.2 | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.189e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 121.8 | 121.8..121.8 | 1 | 0.154 | - | - | 4681.3 | 418.5 | max_abs_diff_vs_ours=5.436e-05, max_rel_diff_vs_ours=2.711e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba1-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 48.4 | 48.4..48.4 | 1 | - | - | - | 103.4 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 25.0 | 25.0..25.0 | 1 | 1.935 | - | - | 2193.0 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.948e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 36.2 | 36.2..36.2 | 1 | 1.337 | - | - | 2189.3 | - | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.57e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16.4 | 16.4..16.4 | 1 | - | - | - | 730.5 | 3248.6 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 13.5 | 13.5..13.5 | 1 | 1.218 | - | - | 3069.0 | 3221.3 | max_abs_diff_vs_ours=1.907e-06, max_rel_diff_vs_ours=6.485e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 12.0 | 12.0..12.0 | 1 | 1.365 | - | - | 3364.3 | 3221.3 | max_abs_diff_vs_ours=1.907e-06, max_rel_diff_vs_ours=6.485e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 12.7 | 12.7..12.7 | 1 | 1.292 | - | - | 4322.1 | 1676.6 | max_abs_diff_vs_ours=0.007144, max_rel_diff_vs_ours=0.002429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 45.3 | 45.3..45.3 | 1 | 0.362 | - | - | 4608.6 | 1676.6 | max_abs_diff_vs_ours=0.007144, max_rel_diff_vs_ours=0.002429 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba2-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 37.0 | 37.0..37.0 | 1 | - | - | - | 121.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 114.3 | 114.3..114.3 | 1 | 0.324 | - | - | 2833.6 | - | max_abs_diff_vs_ours=7.749e-07, max_rel_diff_vs_ours=2.751e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 114.0 | 114.0..114.0 | 1 | 0.325 | - | - | 3407.3 | - | max_abs_diff_vs_ours=7.749e-07, max_rel_diff_vs_ours=2.751e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 93.3 | 93.3..93.3 | 1 | 0.397 | - | - | 2452.9 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 87.3 | 87.3..87.3 | 1 | 0.424 | - | - | 3015.9 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10.8 | 10.8..10.8 | 1 | - | - | - | 2034.4 | 3248.7 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 72.1 | 72.1..72.1 | 1 | 0.150 | - | - | 3063.5 | 285.5 | max_abs_diff_vs_ours=5.364e-07, max_rel_diff_vs_ours=2.347e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 9.6 | 9.6..9.6 | 1 | 1.132 | - | - | 4519.8 | 166.7 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.086e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 73.0 | 73.0..73.0 | 1 | 0.148 | - | - | 4775.7 | 272.4 | max_abs_diff_vs_ours=0.002069, max_rel_diff_vs_ours=0.0009051 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 9.8 | 9.8..9.8 | 1 | 1.106 | - | - | 6256.4 | 134.0 | max_abs_diff_vs_ours=0.001982, max_rel_diff_vs_ours=0.0008672 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba3-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 40.4 | 40.4..40.4 | 1 | - | - | - | 127.7 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 12.6 | 12.6..12.6 | 1 | 3.201 | - | - | 2056.4 | - | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.06e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 4.4 | 4.4..4.4 | 1 | 9.247 | - | - | 2634.9 | - | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.06e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 36.1 | 36.1..36.1 | 1 | 1.119 | - | - | 2051.4 | - | max_abs_diff_vs_ours=0.002148, max_rel_diff_vs_ours=0.0009547 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 40.9 | 40.9..40.9 | 1 | 0.987 | - | - | 2755.2 | - | max_abs_diff_vs_ours=0.002040, max_rel_diff_vs_ours=0.0009065 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-forward.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.5 | 14.5..14.5 | 1 | - | - | - | 2043.6 | 3248.5 | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on cuda failed in round 0 (compile happens here): RuntimeError('self and mat2 must have the same dtype, but got Float and BFloat16')", "event": "error", "stage":) (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 1.6 | 1.6..1.6 | 1 | 9.172 | - | - | 2951.2 | 122.3 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:29:32Z on mojolearn-steward-do-amd, gpu (AMD Instinct Mi325X VF))) |
| torch-eager-bf16 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | 7.719 | - | - | 4136.8 | 139.1 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:29:32Z on mojolearn-steward-do-amd, gpu (AMD Instinct Mi325X VF))) |
| torch-eager-fp32 | torch | gpu | opponent | 1.9 | 1.9..1.9 | 1 | 7.518 | - | - | 2855.2 | 147.3 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:29:32Z on mojolearn-steward-do-amd, gpu (AMD Instinct Mi325X VF))) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, torch-compile-bf16: host not sampled; GPU not sampled

memory, torch-compile-fp32, torch-eager-bf16, torch-eager-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU torch.cuda.max_memory_allocated, reset before the round (caching allocator peak; the context is not in it)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 |

### transformer-infer / gaussian (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-infer.gaussian.shape-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 27.6 | 27.6..27.6 | 1 | - | - | - | 142.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-cpu-compile-bf16 on cpu failed in round 0 (compile happens here): RuntimeError('self and mat2 must have the same dtype, but got Float and BFloat16')", "event": "error", "stag) (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 2.4 | 2.4..2.4 | 1 | 11.706 | - | - | 2555.9 | - | max_abs_diff_vs_ours=7.153e-07, max_rel_diff_vs_ours=1.615e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:30:05Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 2.7 | 2.7..2.7 | 1 | 10.210 | - | - | 2011.8 | - | max_abs_diff_vs_ours=0.001941, max_rel_diff_vs_ours=0.0004383 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:30:05Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 3.3 | 3.3..3.3 | 1 | 8.473 | - | - | 2009.2 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.076e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:30:05Z on mojolearn-steward-do-amd, cpu (AMD EPYC 9575F 64-Core Processor))) |

memory, ours, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-eager-fp32: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

memory, torch-cpu-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 |
|---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 |

## Algorithm expansion

### poisson / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poisson.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 78274.0 | 78274.0..78274.0 | 1 | - | - | - | 831.3 | 2305.2 | finite=True, r2=0.035965, rmse=15.638069 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1405.3 | 1405.3..1405.3 | 1 | 55.699 | - | - | 375.0 | - | finite=True, r2=0.036205, rmse=15.636127 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

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
| mojolearn IDENTICAL | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.8 | 1.8..1.8 | 1 | 0.368 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.665e+06 | 1.665e+06..1.665e+06 | 1 | - | - | - | 1711.9 | 2305.2 | accuracy=0.901150 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28098.7 | 28098.7..28098.7 | 1 | 59.255 | - | - | 1143.5 | - | accuracy=0.910200 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| class_weight | null | null |
| epsilon | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 |
| fit_intercept | true | true |
| l1_ratio | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" |
| loss | "hinge" | "hinge" |
| max_iter | 100 | 100 |
| penalty | "l2" | "l2" |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 10.2 | 10.2..10.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.7 | 3.7..3.7 | 1 | 2.741 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 607156.7 | 607156.7..607156.7 | 1 | - | - | - | 834.7 | 2305.2 | accuracy=0.766410 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8379.1 | 8379.1..8379.1 | 1 | 72.460 | - | - | 266.7 | - | accuracy=0.752520 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| class_weight | null | null |
| epsilon | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 |
| fit_intercept | true | true |
| l1_ratio | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" |
| loss | "hinge" | "hinge" |
| max_iter | 100 | 100 |
| penalty | "l2" | "l2" |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.2 | 8.2..8.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.6 | 0.6..0.6 | 1 | 12.988 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.istella.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.65e+06 | 1.65e+06..1.65e+06 | 1 | - | - | - | 1655.2 | 2305.2 | finite=True, r2=-3.459e+24, rmse=1.554e+12 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 42010.6 | 42010.6..42010.6 | 1 | 39.272 | - | - | 1140.0 | - | finite=True, r2=-2.197e+24, rmse=1.238e+12 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 |
| fit_intercept | true | true |
| l1_ratio | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" |
| loss | "squared_error" | "squared_error" |
| max_iter | 100 | 100 |
| penalty | "l2" | "l2" |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.3 | 2.3..2.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.9 | 3.9..3.9 | 1 | 0.580 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.taxi.rows-full.log`, ran on mojolearn-steward-do-amd

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 574141.9 | 574141.9..574141.9 | 1 | - | - | - | 778.2 | 2305.2 | finite=True, r2=0.868127, rmse=5.783813 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9778.1 | 9778.1..9778.1 | 1 | 58.717 | - | - | 262.7 | - | finite=True, r2=0.880681, rmse=5.501638 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU rocm-smi --showpids VRAM USED for this pid at the round's end (not a peak)

memory, sklearn-cpu: host Linux VmHWM after clear_refs 5 (peak RSS over the round); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | sklearn-cpu |
|---|---||---|---|
| library (source) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 |
| fit_intercept | true | true |
| l1_ratio | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" |
| loss | "squared_error" | "squared_error" |
| max_iter | 100 | 100 |
| penalty | "l2" | "l2" |
| seed | 7 | 7 |
| shuffle | true | true |
| tol | null | null |

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.7 | 0.7..0.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.7 | 0.7..0.7 | 1 | 0.999 | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML and cuVS: CUDA only; no ROCm build is pinned.
- Classical, wave 2, not planned on this vendor: faiss-gpu: the pinned FAISS GPU builds are CUDA; faiss-cpu is the arm on this box.
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
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on ROCm accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on ROCm has no int8 matmul, so ours races alone
- Neural, not planned on this vendor: mamba-ssm-* on mamba*-forward: mamba_ssm publishes no ROCm wheel, and its gfx942 source build (setup.py's HIP path, causal-conv1d's and the Mamba-3 Triton kernels on ROCm) has not been built and checked on the board's AMD box; the Mamba lanes race the torch reference arms there
- Neural, not planned on this vendor: mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its TF32 setting); torch's bf16 arms carry the lower precision

