# mojolearn benchmark board

Generated 2026-09-30T16:55:35Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | apple / metal |
| GPU | Apple M2 Pro |
| GPU driver | macOS 26.6 |
| CPU | Apple M2 Pro (12 logical cores) |
| memory bytes | 34359738368 |
| OS | macOS 26.6 |
| Python | 3.13.15 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-macosx_11_0_arm64.whl, sha256 ee1187c950c29e791c906cb3beb1c45206e4d8534fea897bc839db4505b38548) |
| script commit | a19d159d7d3a7322d0159e6c3482de29c4464109 |
| patch sync | - |
| modes | fast, identical |
| rounds | 1 timed after 1 warm-up, arms interleaved round by round |
| seed | 7 |
| opponent versions | catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0, scikit-learn 1.7.2, torch 2.13.0, umap-learn 0.5.12, pynndescent 0.6.0, numba 0.67.0, statsmodels 0.15.0, faiss-cpu 1.15.1, numpy 2.5.3 |

## How to read this board

- Times are wall milliseconds of the public fit call (trees) or the lane's timed call (classical), median of the timed rounds; min..max beside it.
- `ours IDENTICAL / arm` is our IDENTICAL median divided by that opponent's median; `ours FAST / arm` likewise. Below 1.0 our median time is the lower one, above 1.0 the higher one. A ratio is shown only when both arms completed every round in this run, and only against an opponent: our two modes are never divided by each other here.
- Quality comes from the drivers: FSPEED-ACC for trees (held-out rows), one float64 NumPy function per lane for classical.
- Comparability: trees carry FSPEED-FIT-VERDICT (total leaves within 10% across arms is COMPARABLE); classical carry the clock span (SPAN-ASYMMETRIC names an arm whose clock excludes an upload or a fit that ours includes).
- Classical, wave 2 (`classical2`, tools/bench_board_more.py): the same worker protocol as classical; every lane's parameters, rows, timed span and each unavoidable mismatch with its reason are in the cells' `settings.lane_config`. Quality is one float64 NumPy function per lane over each arm's saved outputs.
- Neural: our IDENTICAL arm only (the neural surface builds no other tier, on any vendor) against torch at every fast setting it supports on this box, one arm each, the setting in the arm name: `torch-eager-fp32` (TF32 off), `torch-compile-fp32` (torch.compile, inductor), `torch-eager-tf32` / `torch-compile-tf32` (NVIDIA CUDA only), `torch-eager-bf16` / `torch-compile-bf16` (bf16 autocast mixed precision). TF32 and bf16 arms are ANOTHER PRECISION than ours; their quality columns show how far. The `*-infer` lanes are the CPU *Inference classes and race `torch-cpu-*` arms. An arm torch cannot run on this box is REFUSED by name in its cell. Every clock is host in, host out, synchronized. Every arm starts from the same parameters and reads the same inputs, so losses and outputs are comparable; `max_abs_diff_vs_ours` / `max_rel_diff_vs_ours` are the arm's output against ours.
- `installed_wheel` confirms our binding loaded from site-packages, not the repo tree.
- Our CPU tier (`mojolearn CPU IDENTICAL`, arm `ours-cpu`): the same public estimator in a worker started under MOJOLEARN_VENDOR=cpu, the wheel's CPU switch (no GPU set loads; the host bindings answer, IDENTICAL only), read back as vendor cpu or refused by name. It races in the same rounds as every arm; `ours CPU / arm` is its median over each opponent's. `bits_equal_vs_ours_identical` compares its output with our GPU IDENTICAL arm's, bit for bit. Our CPU and GPU times are never divided by each other here.
- Memory: `peak host MB` and `peak GPU MB` are the highest per-round peaks over the timed rounds, read outside the clock; each arm's method is listed under its table (host: the resettable peak RSS on Linux, the peak physical footprint on macOS, which holds Metal buffers too; GPU: torch's own counter for torch arms, the driver's per-process figure for the rest, none on Apple).
- Inference: after a race's fit rounds each arm predicts with its own fitted model (no fit retimed), same rows, same output kind, one warm-up then the timed rounds interleaved. Trees: batch `test` (the held-out split) and `large` (1,000,000 training rows, capped at the training rows), host rows in and host predictions out on every arm; each arm's call is printed under its table. Classical: kmeans predict, pca transform, ols predict and svc predict on the eval rows, with the fit's clock span. Ratios are per batch, ours over each opponent.

## Coverage

Races: 1 planned, 114 done, 1 failed, 0 pending. Cells: 385 (HOST-MEMORY 1, REFUSED 36, ok 348).

Inference cells: 138 (REFUSED 9, ok 129).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | ard | istella | r2 (higher is better) | -0.123501 | -0.122912 | - | sklearn-cpu 0.327436 |
| algos | ard | istella | rmse (lower is better) | 0.885416 | 0.885183 | - | sklearn-cpu 0.685058 |
| algos | ard | taxi | r2 (higher is better) | 0.909193 | 0.909193 | - | sklearn-cpu 0.909190 |
| algos | ard | taxi | rmse (lower is better) | 4.799513 | 4.799513 | - | sklearn-cpu 4.799575 |
| algos | bayesian-ridge | istella | r2 (higher is better) | -41688.617698 | -41643.670747 | - | sklearn-cpu -890.186917 |
| algos | bayesian-ridge | istella | rmse (lower is better) | 170.558899 | 170.466932 | - | sklearn-cpu 24.937036 |
| algos | bayesian-ridge | taxi | r2 (higher is better) | 0.908981 | 0.908981 | - | sklearn-cpu 0.908983 |
| algos | bayesian-ridge | taxi | rmse (lower is better) | 4.805108 | 4.805109 | - | sklearn-cpu 4.805052 |
| algos | bisecting-kmeans | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | bisecting-kmeans | istella | silhouette (higher is better) | 0.118345 | 0.118345 | - | sklearn-cpu 0.090241 |
| algos | bisecting-kmeans | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.804388 |
| algos | bisecting-kmeans | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | bisecting-kmeans | taxi | silhouette (higher is better) | 0.155357 | 0.155357 | - | sklearn-cpu 0.133742 |
| algos | bisecting-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.403522 |
| algos | enet-cv | istella | r2 (higher is better) | 0.316583 | 0.316583 | - | sklearn-cpu 0.317292 |
| algos | enet-cv | istella | rmse (lower is better) | 0.690563 | 0.690563 | - | sklearn-cpu 0.690205 |
| algos | enet-cv | taxi | r2 (higher is better) | 0.909002 | 0.909002 | - | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | 4.804540 | 4.804540 | - | sklearn-cpu 4.804486 |
| algos | huber | istella | r2 (higher is better) | -0.008471 | -0.009166 | - | sklearn-cpu -0.010176 |
| algos | huber | istella | rmse (lower is better) | 0.838865 | 0.839154 | - | sklearn-cpu 0.839574 |
| algos | huber | taxi | r2 (higher is better) | 0.900214 | 0.900215 | - | sklearn-cpu 0.900215 |
| algos | huber | taxi | rmse (lower is better) | 5.031205 | 5.031176 | - | sklearn-cpu 5.031163 |
| algos | isotonic | istella | r2 (higher is better) | 0.187985 | 0.187985 | - | sklearn-cpu 0.187985 |
| algos | isotonic | istella | rmse (lower is better) | 0.752735 | 0.752735 | - | sklearn-cpu 0.752735 |
| algos | isotonic | taxi | r2 (higher is better) | 0.897069 | 0.897069 | - | sklearn-cpu 0.897069 |
| algos | isotonic | taxi | rmse (lower is better) | 5.109874 | 5.109874 | - | sklearn-cpu 5.109874 |
| algos | lars | istella | r2 (higher is better) | -1.632561 | -1.360350 | - | sklearn-cpu - |
| algos | lars | istella | rmse (lower is better) | 1.355344 | 1.283360 | - | sklearn-cpu - |
| algos | lars | taxi | r2 (higher is better) | 0.908981 | 0.908981 | - | sklearn-cpu 0.908988 |
| algos | lars | taxi | rmse (lower is better) | 4.805109 | 4.805109 | - | sklearn-cpu 4.804917 |
| algos | lasso-cv | istella | r2 (higher is better) | 0.310329 | 0.310329 | - | sklearn-cpu 0.310837 |
| algos | lasso-cv | istella | rmse (lower is better) | 0.693715 | 0.693715 | - | sklearn-cpu 0.693460 |
| algos | lasso-cv | taxi | r2 (higher is better) | 0.909059 | 0.909059 | - | sklearn-cpu 0.909038 |
| algos | lasso-cv | taxi | rmse (lower is better) | 4.803051 | 4.803051 | - | sklearn-cpu 4.803593 |
| algos | lasso-lars | istella | r2 (higher is better) | 0.310330 | 0.310330 | - | sklearn-cpu 0.311104 |
| algos | lasso-lars | istella | rmse (lower is better) | 0.693715 | 0.693715 | - | sklearn-cpu 0.693326 |
| algos | lasso-lars | taxi | r2 (higher is better) | 0.908996 | 0.908996 | - | sklearn-cpu 0.909003 |
| algos | lasso-lars | taxi | rmse (lower is better) | 4.804698 | 4.804699 | - | sklearn-cpu 4.804527 |
| algos | meanshift | istella | n_clusters | 12 | 12 | - | sklearn-cpu 12 |
| algos | meanshift | istella | silhouette (higher is better) | 0.403452 | 0.403452 | - | sklearn-cpu 0.403452 |
| algos | meanshift | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | meanshift | taxi | n_clusters | 122 | 122 | - | sklearn-cpu 122 |
| algos | meanshift | taxi | silhouette (higher is better) | 0.246592 | 0.246631 | - | sklearn-cpu 0.246631 |
| algos | meanshift | taxi | ari_vs_ours (1 is our partition exactly) | 0.999996 | - | - | sklearn-cpu 1.000000 |
| algos | minibatch-kmeans | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | minibatch-kmeans | istella | silhouette (higher is better) | 0.116696 | 0.116696 | - | sklearn-cpu 0.111849 |
| algos | minibatch-kmeans | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.622446 |
| algos | minibatch-kmeans | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | minibatch-kmeans | taxi | silhouette (higher is better) | 0.138060 | 0.138060 | - | sklearn-cpu 0.165473 |
| algos | minibatch-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.525151 |
| algos | pa-clf | istella | accuracy (higher is better) | 0.903700 | 0.903700 | - | sklearn-cpu 0.890480 |
| algos | pa-clf | taxi | accuracy (higher is better) | 0.583830 | 0.583830 | - | sklearn-cpu 0.744740 |
| algos | pa-reg | istella | r2 (higher is better) | -0.282312 | -0.282312 | - | sklearn-cpu -0.128155 |
| algos | pa-reg | istella | rmse (lower is better) | 0.945926 | 0.945926 | - | sklearn-cpu 0.887248 |
| algos | pa-reg | taxi | r2 (higher is better) | 0.853920 | 0.853920 | - | sklearn-cpu 0.795107 |
| algos | pa-reg | taxi | rmse (lower is better) | 6.087406 | 6.087406 | - | sklearn-cpu 7.209421 |
| algos | perceptron | istella | accuracy (higher is better) | 0.882480 | 0.882480 | - | sklearn-cpu 0.896130 |
| algos | perceptron | taxi | accuracy (higher is better) | 0.465380 | 0.465380 | - | sklearn-cpu 0.750520 |
| algos | quantile | istella | r2 (higher is better) | nan | nan | - | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | nan | nan | - | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | 0.899875 | 0.900039 | - | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | 5.039731 | 5.035615 | - | sklearn-cpu 5.044706 |
| algos | ridge-clf | istella | accuracy (higher is better) | 0.894330 | 0.894330 | - | sklearn-cpu 0.910540 |
| algos | ridge-clf | taxi | accuracy (higher is better) | 0.763570 | 0.763570 | - | sklearn-cpu 0.763580 |
| algos | ridge-cv | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.908988 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.804917 |
| algos | sgd-ocsvm | istella | fraction_flagged | 0.000000 | 0.000000 | - | sklearn-cpu 0.093340 |
| algos | sgd-ocsvm | istella | jaccard_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu 1.000000 |
| algos | sgd-ocsvm | taxi | fraction_flagged | 0.000000 | 0.000000 | - | sklearn-cpu 0.007020 |
| algos | sgd-ocsvm | taxi | jaccard_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu 1.000000 |
| classical | hdbscan | istella | n_clusters | - | - | - | sklearn-cpu 52 |
| classical | hdbscan | istella | noise_fraction | - | - | - | sklearn-cpu 0.252570 |
| classical | hdbscan | istella | rows | - | - | - | sklearn-cpu 100000 |
| classical | hdbscan | taxi | n_clusters | 160 | - | - | sklearn-cpu 161 |
| classical | hdbscan | taxi | noise_fraction | 0.142220 | - | - | sklearn-cpu 0.134620 |
| classical | hdbscan | taxi | rows | 100000 | - | - | sklearn-cpu 100000 |
| classical | kde | istella | mean_log_likelihood (higher is better) | -222.270586 | -222.270586 | - | sklearn-cpu -226.977403 |
| classical | kde | istella | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | -14.826461 | -14.826460 | - | sklearn-cpu -14.826437 |
| classical | kde | taxi | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kmeans | istella | inertia (lower is better) | 6.051e+17 | 6.051e+17 | - | sklearn-cpu 5.959e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | inertia_over_ours | 1.000000 | 1.000000 | - | sklearn-cpu 0.984775; torch-gpu 0.990156 |
| classical | kmeans | istella | n_iter | 33 | 33 | - | sklearn-cpu 31; torch-gpu 63 |
| classical | kmeans | taxi | inertia (lower is better) | 3.093e+08 | 3.093e+08 | - | sklearn-cpu 3.166e+08; torch-gpu 3.06e+08 |
| classical | kmeans | taxi | inertia_over_ours | 1.000000 | 1.000000 | - | sklearn-cpu 1.023705; torch-gpu 0.989329 |
| classical | kmeans | taxi | n_iter | 91 | 91 | - | sklearn-cpu 174; torch-gpu 65 |
| classical | knn | istella | recall_at_k (higher is better) | 0.976613 | 0.976250 | - | sklearn-cpu 1.000000; torch-gpu 0.979973 |
| classical | knn | istella | rows_with_repeated_ids | 0 | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | 0.999754 | 0.999754 | - | sklearn-cpu 1.000000; torch-gpu 0.999738 |
| classical | knn | taxi | rows_with_repeated_ids | 0 | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | 0.321092 | 0.331944 | - | sklearn-cpu 0.001881; torch-gpu - |
| classical | ols | istella | rmse (lower is better) | 0.687544 | 0.682027 | - | sklearn-cpu 0.833655; torch-gpu - |
| classical | ols | taxi | r2 (higher is better) | 0.908838 | 0.908837 | - | sklearn-cpu 0.724848; torch-gpu - |
| classical | ols | taxi | rmse (lower is better) | 4.696444 | 4.696466 | - | sklearn-cpu 8.159214; torch-gpu - |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu - |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | 0.999997 | 0.999996 | - | sklearn-cpu 0.999996; torch-gpu - |
| classical | svc | istella | accuracy (higher is better) | 0.922200 | 0.922200 | - | sklearn-cpu 0.922200 |
| classical | svc | istella | n_support | 2401 | 2400 | - | sklearn-cpu 2400 |
| classical | svc | taxi | accuracy (higher is better) | 0.767500 | 0.767500 | - | sklearn-cpu 0.767500 |
| classical | svc | taxi | n_support | 5533 | 5527 | - | sklearn-cpu 5672 |
| classical2 | agglomerative | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | agglomerative | istella | silhouette (higher is better) | 0.716728 | 0.716728 | - | sklearn-cpu 0.716728 |
| classical2 | agglomerative | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical2 | agglomerative | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | agglomerative | taxi | silhouette (higher is better) | 0.685524 | 0.685524 | - | sklearn-cpu 0.685524 |
| classical2 | agglomerative | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical2 | arima | synthetic | forecast_rmse (lower is better) | 1.515540 | 1.515518 | - | statsmodels-cpu 1.515423 |
| classical2 | arima | synthetic | insample_rmse (lower is better) | 0.999341 | 0.999342 | - | statsmodels-cpu 0.999338 |
| classical2 | arima | synthetic | mean_aic (lower is better) | 5680.971687 | 5680.976967 | - | statsmodels-cpu 5680.957160 |
| classical2 | arima | synthetic | mean_llf (higher is better) | -2836.485844 | -2836.488483 | - | statsmodels-cpu -2836.478580 |
| classical2 | elasticnet | istella | r2 (higher is better) | 0.261554 | 0.260922 | - | sklearn-cpu 0.260922 |
| classical2 | elasticnet | istella | rmse (lower is better) | 0.717827 | 0.718134 | - | sklearn-cpu 0.718134 |
| classical2 | elasticnet | taxi | r2 (higher is better) | 0.907378 | 0.907378 | - | sklearn-cpu 0.907378 |
| classical2 | elasticnet | taxi | rmse (lower is better) | 4.847224 | 4.847224 | - | sklearn-cpu 4.847223 |
| classical2 | ets | synthetic | forecast_rmse (lower is better) | 0.984473 | 0.984392 | - | statsmodels-cpu 0.984418 |
| classical2 | ets | synthetic | insample_rmse (lower is better) | 0.990971 | 0.990971 | - | statsmodels-cpu 0.991812 |
| classical2 | gmm | istella | bic (lower is better) | -3.851e+07 | -3.851e+07 | - | sklearn-cpu -3.901e+07 |
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | 200.794408 | 200.794403 | - | sklearn-cpu 200.776340 |
| classical2 | gmm | istella | n_iter | 24 | 24 | - | sklearn-cpu 30 |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.67e+06 | - | sklearn-cpu - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.861940 | - | sklearn-cpu - |
| classical2 | gmm | taxi | n_iter | - | 32 | - | sklearn-cpu - |
| classical2 | gpc | istella | accuracy (higher is better) | 0.901333 | 0.901333 | - | sklearn-cpu 0.901333 |
| classical2 | gpc | istella | logloss (lower is better) | 0.232594 | 0.232590 | - | sklearn-cpu 0.232597 |
| classical2 | gpc | istella | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | gpc | taxi | accuracy (higher is better) | 0.761000 | 0.761000 | - | sklearn-cpu 0.761000 |
| classical2 | gpc | taxi | logloss (lower is better) | 0.541358 | 0.541286 | - | sklearn-cpu 0.541358 |
| classical2 | gpc | taxi | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | gpr | istella | mean_log_predictive_density (higher is better) | -9.285405 | -9.285754 | - | sklearn-cpu -9.287148 |
| classical2 | gpr | istella | r2 (higher is better) | 0.235383 | 0.235346 | - | sklearn-cpu 0.235368 |
| classical2 | gpr | istella | rmse (lower is better) | 0.760421 | 0.760439 | - | sklearn-cpu 0.760428 |
| classical2 | gpr | taxi | mean_log_predictive_density (higher is better) | -311.471466 | -311.458394 | - | sklearn-cpu -311.539594 |
| classical2 | gpr | taxi | r2 (higher is better) | 0.889629 | 0.889630 | - | sklearn-cpu 0.889629 |
| classical2 | gpr | taxi | rmse (lower is better) | 5.041669 | 5.041639 | - | sklearn-cpu 5.041653 |
| classical2 | kernel-ridge | istella | r2 (higher is better) | 0.385427 | 0.385427 | - | sklearn-cpu 0.385427 |
| classical2 | kernel-ridge | istella | rmse (lower is better) | 0.646407 | 0.646407 | - | sklearn-cpu 0.646407 |
| classical2 | kernel-ridge | taxi | r2 (higher is better) | 0.726543 | 0.726543 | - | sklearn-cpu 0.726542 |
| classical2 | kernel-ridge | taxi | rmse (lower is better) | 8.330374 | 8.330373 | - | sklearn-cpu 8.330380 |
| classical2 | knn-clf | istella | accuracy (higher is better) | 0.926250 | 0.926250 | - | sklearn-cpu 0.926250 |
| classical2 | knn-clf | taxi | accuracy (higher is better) | 0.741750 | 0.741750 | - | sklearn-cpu 0.741750 |
| classical2 | knn-reg | istella | r2 (higher is better) | 0.418145 | 0.418145 | - | sklearn-cpu 0.418145 |
| classical2 | knn-reg | istella | rmse (lower is better) | 0.625388 | 0.625388 | - | sklearn-cpu 0.625388 |
| classical2 | knn-reg | taxi | r2 (higher is better) | 0.937323 | 0.937323 | - | sklearn-cpu 0.937323 |
| classical2 | knn-reg | taxi | rmse (lower is better) | 3.842028 | 3.842028 | - | sklearn-cpu 3.842028 |
| classical2 | lasso | istella | r2 (higher is better) | 0.310472 | 0.310837 | - | sklearn-cpu 0.310837 |
| classical2 | lasso | istella | rmse (lower is better) | 0.693643 | 0.693460 | - | sklearn-cpu 0.693460 |
| classical2 | lasso | taxi | r2 (higher is better) | 0.908995 | 0.908995 | - | sklearn-cpu 0.908995 |
| classical2 | lasso | taxi | rmse (lower is better) | 4.804745 | 4.804745 | - | sklearn-cpu 4.804745 |
| classical2 | linearsvc | istella | accuracy (higher is better) | 0.923230 | 0.923480 | - | sklearn-cpu 0.923540 |
| classical2 | linearsvc | taxi | accuracy (higher is better) | 0.763330 | 0.763330 | - | sklearn-cpu 0.763570 |
| classical2 | linearsvr | istella | r2 (higher is better) | -0.106754 | -0.106754 | - | sklearn-cpu -0.025729 |
| classical2 | linearsvr | istella | rmse (lower is better) | 0.878792 | 0.878792 | - | sklearn-cpu 0.846012 |
| classical2 | linearsvr | taxi | r2 (higher is better) | 0.899814 | 0.899813 | - | sklearn-cpu 0.899803 |
| classical2 | linearsvr | taxi | rmse (lower is better) | 5.041262 | 5.041302 | - | sklearn-cpu 5.041552 |
| classical2 | logreg | istella | accuracy (higher is better) | 0.924570 | 0.924540 | - | sklearn-cpu 0.924470 |
| classical2 | logreg | istella | logloss (lower is better) | 0.181257 | 0.181245 | - | sklearn-cpu 0.181337 |
| classical2 | logreg | istella | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | 0.763350 | 0.763340 | - | sklearn-cpu 0.763320 |
| classical2 | logreg | taxi | logloss (lower is better) | 0.538986 | 0.538984 | - | sklearn-cpu 0.538980 |
| classical2 | logreg | taxi | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | nystroem | istella | kernel_rel_error (lower is better) | - | - | - | sklearn-cpu 0.038958 |
| classical2 | nystroem | taxi | kernel_rel_error (lower is better) | - | - | - | sklearn-cpu 0.044370 |
| classical2 | rbf-sampler | istella | kernel_rel_error (lower is better) | 0.141980 | 0.141980 | - | sklearn-cpu 0.137405 |
| classical2 | rbf-sampler | taxi | kernel_rel_error (lower is better) | 0.108549 | 0.108549 | - | sklearn-cpu 0.083775 |
| classical2 | ridge | istella | r2 (higher is better) | 0.320451 | 0.328682 | - | sklearn-cpu 0.328676 |
| classical2 | ridge | istella | rmse (lower is better) | 0.688606 | 0.684423 | - | sklearn-cpu 0.684426 |
| classical2 | ridge | taxi | r2 (higher is better) | 0.908983 | 0.908983 | - | sklearn-cpu 0.908988 |
| classical2 | ridge | taxi | rmse (lower is better) | 4.805048 | 4.805042 | - | sklearn-cpu 4.804916 |
| classical2 | spectral-embedding | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.823640 | 0.799378 | - | sklearn-cpu 0.812688 |
| classical2 | spectral-embedding | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.895236 | 0.884889 | - | sklearn-cpu 0.898011 |
| classical2 | spectral | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | spectral | istella | silhouette (higher is better) | 0.147668 | 0.147668 | - | sklearn-cpu 0.147699 |
| classical2 | spectral | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.999826 |
| classical2 | spectral | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | spectral | taxi | silhouette (higher is better) | 0.039888 | 0.039910 | - | sklearn-cpu 0.089894 |
| classical2 | spectral | taxi | ari_vs_ours (1 is our partition exactly) | 0.997088 | - | - | sklearn-cpu 0.582313 |
| classical2 | svr | istella | r2 (higher is better) | 0.318257 | 0.318258 | - | sklearn-cpu 0.318248 |
| classical2 | svr | istella | rmse (lower is better) | 0.680816 | 0.680816 | - | sklearn-cpu 0.680821 |
| classical2 | svr | taxi | r2 (higher is better) | 0.767550 | 0.767551 | - | sklearn-cpu 0.767550 |
| classical2 | svr | taxi | rmse (lower is better) | 7.680415 | 7.680395 | - | sklearn-cpu 7.680414 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | 0.999992 | 0.999992 | - | sklearn-cpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | 0.002554 | 0.002554 | - | sklearn-cpu 0.000122 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | 0.999965 | 0.999965 | - | sklearn-cpu 0.999965 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | 0.003257 | 0.003257 | - | sklearn-cpu 0.003257 |
| classical2 | umap | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.981706 | 0.979906 | - | umap-learn-cpu 0.978822; umap-learn-cpu-unseeded 0.977512 |
| classical2 | umap | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.991729 | 0.990480 | - | umap-learn-cpu 0.989525; umap-learn-cpu-unseeded 0.990395 |
| neural | gemm-bf16 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 1.155e-07 | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-bf16 | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 0.999023; torch-compile-bf16 0.999023 |
| neural | gemm-bf16 | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-int8 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 0.000000 | - | - |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | - | torch-eager-fp32 2.843e-06; torch-compile-fp32 2.843e-06; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 0.001038; torch-compile-fp32 0.001038; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.866e-06; torch-compile-fp32 2.866e-06; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | - | torch-eager-fp32 9.018733; torch-compile-fp32 9.018733; torch-eager-bf16 9.018647; torch-compile-bf16 9.018634 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.241e-06; torch-compile-fp32 1.237e-06; torch-eager-bf16 0.006710; torch-compile-bf16 0.006710 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.703e-06; torch-compile-fp32 1.698e-06; torch-eager-bf16 0.009211; torch-compile-bf16 0.009211 |
| neural | lm-host-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.017858 | - | torch-cpu-eager-fp32 9.017857; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 9.017766; torch-cpu-compile-bf16 - |
| neural | lm-host-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.367768 | - | torch-cpu-eager-fp32 8.367766; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 8.368875; torch-cpu-compile-bf16 - |
| neural | lm-host-train-step | bytes | steps | - | 2 | - | torch-cpu-eager-fp32 2; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 2; torch-cpu-compile-bf16 - |
| neural | lm-host-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-eager-fp32 9.537e-07; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 9.155e-05; torch-cpu-compile-bf16 - |
| neural | lm-host-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-eager-fp32 1.907e-06; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.001106; torch-cpu-compile-bf16 - |
| neural | lm-infer | bytes | mean_nll (lower is better) | - | 9.017857 | - | torch-cpu-eager-fp32 9.017857; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 9.017766; torch-cpu-compile-bf16 - |
| neural | lm-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 9.239e-07; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.006131; torch-cpu-compile-bf16 - |
| neural | lm-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.268e-06; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.008417; torch-cpu-compile-bf16 - |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | - | torch-eager-fp32 9.018733; torch-compile-fp32 9.018734; torch-eager-bf16 9.022064; torch-compile-bf16 9.018402 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.422411 | - | torch-eager-fp32 8.422411; torch-compile-fp32 8.422413; torch-eager-bf16 8.418777; torch-compile-bf16 8.420258 |
| neural | lm-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | lm-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 9.537e-07; torch-eager-bf16 0.003331; torch-compile-bf16 0.0003309 |
| neural | lm-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 1.907e-06; torch-eager-bf16 0.003633; torch-compile-bf16 0.002153 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-eager-bf16 5.15e-05 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.189e-07; torch-eager-bf16 2.568e-05 |
| neural | mamba1-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-07; torch-cpu-eager-bf16 5.15e-05 |
| neural | mamba1-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 5.948e-08; torch-cpu-eager-bf16 2.57e-05 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.146e-06; torch-compile-fp32 2.146e-06; torch-eager-bf16 0.007544; torch-compile-bf16 0.007544 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 7.295e-07; torch-compile-fp32 7.295e-07; torch-eager-bf16 0.002565; torch-compile-bf16 0.002565 |
| neural | mamba2-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-06; torch-cpu-compile-fp32 1.192e-06; torch-cpu-eager-bf16 0.005920; torch-cpu-compile-bf16 0.005920 |
| neural | mamba2-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 4.233e-07; torch-cpu-compile-fp32 4.233e-07; torch-cpu-eager-bf16 0.002102; torch-cpu-compile-bf16 0.002102 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 7.153e-07; torch-compile-fp32 -; torch-eager-bf16 0.002095; torch-compile-bf16 - |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 3.129e-07; torch-compile-fp32 -; torch-eager-bf16 0.0009164; torch-compile-bf16 - |
| neural | mamba3-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 4.768e-07; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.002148; torch-cpu-compile-bf16 - |
| neural | mamba3-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.119e-07; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.0009547; torch-cpu-compile-bf16 - |
| neural | mlp-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 0.000000; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.004285; torch-cpu-compile-bf16 - |
| neural | mlp-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 0.000000; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.003956; torch-cpu-compile-bf16 - |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | - | torch-eager-fp32 1.160401; torch-compile-fp32 1.160401; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | - | torch-eager-fp32 1.123361; torch-compile-fp32 1.123361; torch-eager-bf16 1.123461; torch-compile-bf16 1.123462 |
| neural | mlp-train-step | gaussian | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 9.656e-05; torch-compile-bf16 9.656e-05 |
| neural | mlp-train-step | gaussian | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 0.0001006; torch-compile-bf16 0.0001007 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-eager-fp32 5.635910; torch-compile-fp32 -; torch-eager-bf16 5.636023; torch-compile-bf16 - |
| neural | samba-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.623e-06; torch-compile-fp32 -; torch-eager-bf16 0.016692; torch-compile-bf16 - |
| neural | samba-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.474e-06; torch-compile-fp32 -; torch-eager-bf16 0.009384; torch-compile-bf16 - |
| neural | samba-infer | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-cpu-eager-fp32 5.635910; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 5.636032; torch-cpu-compile-bf16 - |
| neural | samba-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 2.682e-06; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.018826; torch-cpu-compile-bf16 - |
| neural | samba-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.508e-06; torch-cpu-compile-fp32 -; torch-cpu-eager-bf16 0.010584; torch-cpu-compile-bf16 - |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | - | torch-eager-fp32 5.635910; torch-compile-fp32 -; torch-eager-bf16 5.636022; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | - | torch-eager-fp32 4.833934; torch-compile-fp32 -; torch-eager-bf16 4.833807; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 -; torch-eager-bf16 2; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 -; torch-eager-bf16 0.0001116; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 -; torch-eager-bf16 0.0001273; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.001819; torch-compile-bf16 0.001819 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.02e-07; torch-compile-fp32 1.02e-07; torch-eager-bf16 0.0003893; torch-compile-bf16 0.0003893 |
| neural | transformer-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 4.768e-07; torch-cpu-compile-fp32 9.775e-06; torch-cpu-eager-bf16 0.001941; torch-cpu-compile-bf16 0.001819 |
| neural | transformer-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.076e-07; torch-cpu-compile-fp32 2.207e-06; torch-cpu-eager-bf16 0.0004383; torch-cpu-compile-bf16 0.0004107 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | ard | istella | Xq | - | 13.2 | 13.5 | - | - | - | sklearn-cpu 5.9 ms (IDENTICAL/arm 2.287) |
| algos | ard | taxi | Xq | - | 1.4 | 1.4 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 3.719) |
| algos | bayesian-ridge | istella | Xq | - | 15.4 | 14.2 | - | - | - | sklearn-cpu 6.0 ms (IDENTICAL/arm 2.378) |
| algos | bayesian-ridge | taxi | Xq | - | 2.2 | 2.0 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 4.637) |
| algos | bisecting-kmeans | istella | Xq | - | 43.5 | 44.4 | - | - | - | sklearn-cpu 58.3 ms (IDENTICAL/arm 0.761) |
| algos | bisecting-kmeans | taxi | Xq | - | 4.2 | 3.9 | - | - | - | sklearn-cpu 12.8 ms (IDENTICAL/arm 0.304) |
| algos | enet-cv | istella | Xq | - | 14.1 | 14.1 | - | - | - | sklearn-cpu 5.9 ms (IDENTICAL/arm 2.377) |
| algos | enet-cv | taxi | Xq | - | 2.8 | 2.0 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 5.434) |
| algos | huber | istella | Xq | - | 14.2 | 13.5 | - | - | - | sklearn-cpu 22.3 ms (IDENTICAL/arm 0.603) |
| algos | huber | taxi | Xq | - | 2.2 | 2.2 | - | - | - | sklearn-cpu 1.4 ms (IDENTICAL/arm 1.565) |
| algos | isotonic | istella | Xq | - | 8.2 | 8.1 | - | - | - | sklearn-cpu 2.8 ms (IDENTICAL/arm 2.913) |
| algos | isotonic | taxi | Xq | - | 38.2 | 35.5 | - | - | - | sklearn-cpu 5.3 ms (IDENTICAL/arm 6.724) |
| algos | lars | istella | Xq | - | 14.3 | 13.7 | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | lars | taxi | Xq | - | 2.0 | 2.1 | - | - | - | sklearn-cpu 0.5 ms (IDENTICAL/arm 4.572) |
| algos | lasso-cv | istella | Xq | - | 15.6 | 26.1 | - | - | - | sklearn-cpu 5.9 ms (IDENTICAL/arm 4.443) |
| algos | lasso-cv | taxi | Xq | - | 2.8 | 3.0 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 7.140) |
| algos | lasso-lars | istella | Xq | - | 13.9 | 14.2 | - | - | - | sklearn-cpu 6.0 ms (IDENTICAL/arm 2.387) |
| algos | lasso-lars | taxi | Xq | - | 2.1 | 2.0 | - | - | - | sklearn-cpu 0.5 ms (IDENTICAL/arm 4.352) |
| algos | meanshift | istella | Xq | - | 42.0 | 43.6 | - | - | - | sklearn-cpu 10.6 ms (IDENTICAL/arm 4.126) |
| algos | meanshift | taxi | Xq | - | 9.0 | 8.6 | - | - | - | sklearn-cpu 25.9 ms (IDENTICAL/arm 0.333) |
| algos | minibatch-kmeans | istella | Xq | - | 54.5 | 54.8 | - | - | - | sklearn-cpu 6.6 ms (IDENTICAL/arm 8.265) |
| algos | minibatch-kmeans | taxi | Xq | - | 2.9 | 3.2 | - | - | - | sklearn-cpu 0.9 ms (IDENTICAL/arm 3.459) |
| algos | pa-clf | istella | Xq | - | 23.2 | 22.8 | - | - | - | sklearn-cpu 6.6 ms (IDENTICAL/arm 3.438) |
| algos | pa-clf | taxi | Xq | - | 11.6 | 11.4 | - | - | - | sklearn-cpu 1.0 ms (IDENTICAL/arm 10.925) |
| algos | pa-reg | istella | Xq | - | 14.7 | 14.8 | - | - | - | sklearn-cpu 6.1 ms (IDENTICAL/arm 2.429) |
| algos | pa-reg | taxi | Xq | - | 2.2 | 2.4 | - | - | - | sklearn-cpu 0.7 ms (IDENTICAL/arm 3.201) |
| algos | perceptron | istella | Xq | - | 25.8 | 22.8 | - | - | - | sklearn-cpu 6.5 ms (IDENTICAL/arm 3.482) |
| algos | perceptron | taxi | Xq | - | 11.6 | 11.5 | - | - | - | sklearn-cpu 1.1 ms (IDENTICAL/arm 10.286) |
| algos | quantile | istella | Xq | - | 13.2 | 13.3 | - | - | - | sklearn-cpu 22.6 ms (IDENTICAL/arm 0.587) |
| algos | quantile | taxi | Xq | - | 1.6 | 2.2 | - | - | - | sklearn-cpu 1.6 ms (IDENTICAL/arm 1.367) |
| algos | ridge-clf | istella | Xq | - | 22.7 | 22.1 | - | - | - | sklearn-cpu 6.4 ms (IDENTICAL/arm 3.436) |
| algos | ridge-clf | taxi | Xq | - | 11.7 | 12.7 | - | - | - | sklearn-cpu 1.1 ms (IDENTICAL/arm 11.576) |
| algos | ridge-cv | istella | Xq | - | - | - | - | - | - | sklearn-cpu 5.8 ms (IDENTICAL/arm -) |
| algos | ridge-cv | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 0.6 ms (IDENTICAL/arm -) |
| algos | sgd-ocsvm | istella | Xq | - | 32.9 | 23.8 | - | - | - | sklearn-cpu 6.2 ms (IDENTICAL/arm 3.838) |
| algos | sgd-ocsvm | taxi | Xq | - | 8.3 | 7.9 | - | - | - | sklearn-cpu 0.8 ms (IDENTICAL/arm 10.318) |
| classical | kmeans | istella | Xq | 500000 | 300.3 | 169.8 | yes | - | - | sklearn-cpu 66.9 ms (IDENTICAL/arm 2.539); torch-gpu 69.8 ms (IDENTICAL/arm 2.433) |
| classical | kmeans | taxi | Xq | 500000 | 28.6 | 33.4 | yes | - | - | sklearn-cpu 14.9 ms (IDENTICAL/arm 2.249); torch-gpu 53.2 ms (IDENTICAL/arm 0.628) |
| classical | ols | istella | Xq | 500000 | 83.8 | 109.4 | no | - | - | sklearn-cpu 63.7 ms (IDENTICAL/arm 1.718); torch-gpu - ms (IDENTICAL/arm -) |
| classical | ols | taxi | Xq | 500000 | 24.4 | 23.5 | no | - | - | sklearn-cpu 6.6 ms (IDENTICAL/arm 3.586); torch-gpu - ms (IDENTICAL/arm -) |
| classical | pca | istella | Xq | 500000 | 202.9 | 223.9 | no | - | - | sklearn-cpu 77.9 ms (IDENTICAL/arm 2.874); torch-gpu - ms (IDENTICAL/arm -) |
| classical | pca | taxi | Xq | 500000 | 49.0 | 56.1 | no | - | - | sklearn-cpu 17.1 ms (IDENTICAL/arm 3.275); torch-gpu - ms (IDENTICAL/arm -) |
| classical | svc | istella | Xq | 10000 | 181.6 | 90.0 | yes | - | - | sklearn-cpu 2095.6 ms (IDENTICAL/arm 0.043) |
| classical | svc | taxi | Xq | 10000 | 35.3 | 78.4 | yes | - | - | sklearn-cpu 1565.4 ms (IDENTICAL/arm 0.050) |

## Classical

### dbscan / taxi (rows full, shape 1000000x11)

race: failed, driver rc 0, log `logs/classical.dbscan.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('dbscan: the ball-cover neighbourhood has -2138378046 edges in one batch, which does not fit the int32 CSR this implementation uses. cuML requires int64 labels for RBC (runner.cuh) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('dbscan: the ball-cover neighbourhood has -2138378046 edges in one batch, which does not fit the int32 CSR this implementation uses. cuML requires int64 labels for RBC (runner.cuh) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | HOST-MEMORY(killed at 30.6 GB: the driver's process tree held 32.9 GB, over 90% of the box's 34.4 GB) (measured this run) |

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "rbc" | "rbc" | "brute" |
| eps | 3.0 | 3.0 | 3.0 |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_samples | 2 | 2 | 2 |
| p | - | - | null |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu algorithm: an exact eps search on every arm: ours 'rbc' (its default), scikit-learn has no 'rbc' and runs 'brute' (the cuML benchmark's cpu_args)

### hdbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| sklearn-cpu | scikit-learn | cpu | opponent | 852655.7 | 852655.7..852655.7 | 1 | - | - | - | 1368.1 | - | n_clusters=52, noise_fraction=0.252570, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | - | - | "auto" |
| allow_single_cluster | false | false | false |
| alpha | 1.0 | 1.0 | 1.0 |
| cluster_selection_epsilon | 0.0 | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" | "eom" |
| leaf_size | - | - | 40 |
| max_cluster_size | 0 | 0 | null |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 | 100 |
| min_samples | 10 | 10 | 10 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu max_cluster_size: no limit on every arm: ours and cuML spell it 0, scikit-learn None

### hdbscan / taxi (rows full, shape 1000000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| mojolearn FAST | mojolearn | gpu | fast | 2443.9 | 2443.9..2443.9 | 1 | - | - | - | 229.6 | - | n_clusters=160, noise_fraction=0.142220, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 42033.3 | 42033.3..42033.3 | 1 | - | 0.058 | - | 235.5 | - | n_clusters=161, noise_fraction=0.134620, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows: the dbscan block's first 100,000 rows. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)

mismatch: scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo at its default

config: cuML benchmark (RAPIDS), HDBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | - | - | "auto" |
| allow_single_cluster | false | false | false |
| alpha | 1.0 | 1.0 | 1.0 |
| cluster_selection_epsilon | 0.0 | 0.0 | 0.0 |
| cluster_selection_method | "eom" | "eom" | "eom" |
| leaf_size | - | - | 40 |
| max_cluster_size | 0 | 0 | null |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_cluster_size | 100 | 100 | 100 |
| min_samples | 10 | 10 | 10 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: sklearn-cpu max_cluster_size: no limit on every arm: ours and cuML spell it 0, scikit-learn None

### kde / istella (rows full, shape 100000x220)

race: done, driver rc 0, log `logs/classical.kde.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 573.7 | 573.7..573.7 | 1 | - | - | - | 1001.8 | - | mean_log_likelihood=-222.270586, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2222.5 | 2222.5..2222.5 | 1 | - | - | - | 234.7 | - | mean_log_likelihood=-222.270586, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 62876.9 | 62876.9..62876.9 | 1 | 0.009 | 0.035 | - | 408.1 | - | mean_log_likelihood=-226.977403, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| atol | 0.0 | 0.0 | 0.0 |
| bandwidth | 1.0 | 1.0 | 1.0 |
| breadth_first | true | true | true |
| kernel | "gaussian" | "gaussian" | "gaussian" |
| leaf_size | 40 | 40 | 40 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| rtol | 0.0 | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### kde / taxi (rows full, shape 100000x11)

race: done, driver rc 0, log `logs/classical.kde.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 262.1 | 262.1..262.1 | 1 | - | - | - | 832.9 | - | mean_log_likelihood=-14.826460, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 72.8 | 72.8..72.8 | 1 | - | - | - | 64.9 | - | mean_log_likelihood=-14.826461, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8497.9 | 8497.9..8497.9 | 1 | 0.031 | 0.009 | - | 147.6 | - | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows: kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact density)

mismatch: cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)

config: cuML benchmark (RAPIDS), KernelDensity (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| atol | 0.0 | 0.0 | 0.0 |
| bandwidth | 1.0 | 1.0 | 1.0 |
| breadth_first | true | true | true |
| kernel | "gaussian" | "gaussian" | "gaussian" |
| leaf_size | 40 | 40 | 40 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| rtol | 0.0 | 0.0 | 0.0 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### kmeans / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7376.3 | 7376.3..7376.3 | 1 | - | - | - | 4059.9 | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7879.5 | 7879.5..7879.5 | 1 | - | - | - | 4052.3 | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5074.3 | 5074.3..5074.3 | 1 | 1.454 | 1.553 | - | 5745.9 | - | inertia=5.959e+17, inertia_over_ours=0.984775, n_iter=31 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 28126.1 | 28126.1..28126.1 | 1 | 0.262 | 0.280 | - | 9251.8 | 6912.6 | inertia=5.991e+17, inertia_over_ours=0.990156, n_iter=63 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | - | - | "lloyd" | - |
| init | "k-means++" | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 300 | 300 | 300 | 300 |
| metric | "euclidean" | "euclidean" | - | "euclidean" |
| n_clusters | 8 | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 | - | - |
| seed | 7 | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 169.8 | 169.8..169.8 | 1 | - | - | - | eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 300.3 | 300.3..300.3 | 1 | - | - | - | agreement_vs_ours=1.000000, bits_equal_vs_ours=True, eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 66.9 | 66.9..66.9 | 1 | 2.539 | 4.490 | - | agreement_vs_ours=0.000000, bits_equal_vs_ours=False, eval_inertia=1.412e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 69.8 | 69.8..69.8 | 1 | 2.433 | 4.302 | - | agreement_vs_ours=0.001558, bits_equal_vs_ours=False, eval_inertia=1.402e+17, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3230.2 | 3230.2..3230.2 | 1 | - | - | - | 680.0 | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2992.7 | 2992.7..2992.7 | 1 | - | - | - | 679.0 | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3981.4 | 3981.4..3981.4 | 1 | 0.811 | 0.752 | - | 407.0 | - | inertia=3.166e+08, inertia_over_ours=1.023705, n_iter=174 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 7123.2 | 7123.2..7123.2 | 1 | 0.453 | 0.420 | - | 1628.4 | 1208.6 | inertia=3.06e+08, inertia_over_ours=0.989329, n_iter=65 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows: big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

mismatch: k-means++: each library draws its own start from its own generator seeded 7, so the starts differ; scikit-learn greedy k-means++ (2 + log k candidates per center), ours and cuML the sequential k-means++ of cuML (oversampling_factor=0), torch-gpu scikit-learn's greedy rule written out

mismatch: tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library applies it through its own convergence test; torch-gpu stops by scikit-learn's (center shift <= tol x mean feature variance). n_iter is in every quality cell

mismatch: algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter (Lloyd); torch-gpu is written as Lloyd

config: cuML benchmark (RAPIDS), KMeans (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | - | - | "lloyd" | - |
| init | "k-means++" | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 300 | 300 | 300 | 300 |
| metric | "euclidean" | "euclidean" | - | "euclidean" |
| n_clusters | 8 | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 | 1 |
| oversampling_factor | 0.0 | 0.0 | - | - |
| seed | 7 | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 | 1e-07 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 33.4 | 33.4..33.4 | 1 | - | - | - | eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 28.6 | 28.6..28.6 | 1 | - | - | - | agreement_vs_ours=1.000000, bits_equal_vs_ours=True, eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 14.9 | 14.9..14.9 | 1 | 2.249 | 1.924 | - | agreement_vs_ours=0.021912, bits_equal_vs_ours=False, eval_inertia=5.046e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 53.2 | 53.2..53.2 | 1 | 0.628 | 0.537 | - | agreement_vs_ours=0.020702, bits_equal_vs_ours=False, eval_inertia=4.571e+07, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### knn / istella (rows full, shape 400000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5407.6 | 5407.6..5407.6 | 1 | - | - | - | 1375.2 | - | recall_at_k=0.976250, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6671.2 | 6671.2..6671.2 | 1 | - | - | - | 1337.3 | - | recall_at_k=0.976613, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1391.0 | 1391.0..1391.0 | 1 | 3.887 | 4.796 | - | 507.7 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2540.3 | 2540.3..2540.3 | 1 | 2.129 | 2.626 | - | 9403.5 | 8892.4 | recall_at_k=0.979973, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 | - |
| metric | "euclidean" | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 | 64 | 64 |
| p | 2 | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | 7 |

### knn / taxi (rows full, shape 400000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5939.3 | 5939.3..5939.3 | 1 | - | - | - | 411.4 | - | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 809.3 | 809.3..809.3 | 1 | - | - | - | 690.0 | - | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 823.1 | 823.1..823.1 | 1 | 7.216 | 0.983 | - | 174.4 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2418.0 | 2418.0..2418.0 | 1 | 2.456 | 0.335 | - | 9075.2 | 8886.4 | recall_at_k=0.999738, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows: knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

mismatch: seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)

mismatch: query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), NearestNeighbors (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 | - |
| metric | "euclidean" | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 64 | 64 | 64 | 64 |
| p | 2 | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | 7 |

### ols / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.ols.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4609.9 | 4609.9..4609.9 | 1 | - | - | - | 9082.9 | - | finite=True, r2=0.331944, rmse=0.682027 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4985.2 | 4985.2..4985.2 | 1 | - | - | - | 9082.8 | - | finite=True, r2=0.321092, rmse=0.687544 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5117.0 | 5117.0..5117.0 | 1 | 0.901 | 0.974 | - | 5723.9 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| fit_intercept | true | true | true | true |
| positive | - | - | false | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | 7 |
| tol | - | - | 1e-06 | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 109.4 | 109.4..109.4 | 1 | - | - | - | predict_max_rel_err_own_fp64=9.581e-07, r2_eval=0.331944, rmse_eval=0.682027 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 83.8 | 83.8..83.8 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=2.693979, predict_max_rel_err_own_fp64=8.387e-07, r2_eval=0.321092, rmse_eval=0.687544 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 63.7 | 63.7..63.7 | 1 | 1.718 | 1.315 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=7.010309, predict_max_rel_err_own_fp64=1.18e-07, r2_eval=0.001881, rmse_eval=0.833655 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: predict(Xq)

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 151.6 | 151.6..151.6 | 1 | - | - | - | 797.5 | - | finite=True, r2=0.908837, rmse=4.696466 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 137.3 | 137.3..137.3 | 1 | - | - | - | 797.6 | - | finite=True, r2=0.908838, rmse=4.696444 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 306.3 | 306.3..306.3 | 1 | 0.495 | 0.448 | - | 376.4 | - | finite=True, r2=0.724848, rmse=8.159214 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: fit_intercept=True. Rows: big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)

mismatch: solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff

mismatch: scikit-learn positive=False and copy_X=True: parameters ours does not have

config: cuML benchmark (RAPIDS), LinearRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| fit_intercept | true | true | true | true |
| positive | - | - | false | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" | 7 |
| tol | - | - | 1e-06 | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 23.5 | 23.5..23.5 | 1 | - | - | - | predict_max_rel_err_own_fp64=8.387e-08, r2_eval=0.908837, rmse_eval=4.696466 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 24.4 | 24.4..24.4 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.026733, predict_max_rel_err_own_fp64=1.15e-07, r2_eval=0.908838, rmse_eval=4.696444 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 6.6 | 6.6..6.6 | 1 | 3.586 | 3.718 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=197.959106, predict_max_rel_err_own_fp64=1.016e-07, r2_eval=0.724848, rmse_eval=8.159213 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: predict(Xq)

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3817.4 | 3817.4..3817.4 | 1 | - | - | - | 7349.3 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4125.1 | 4125.1..4125.1 | 1 | - | - | - | 7348.3 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 285.2 | 285.2..285.2 | 1 | 13.386 | 14.465 | - | 2280.2 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::_linalg_eigh.eigenvalues' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on http) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| n_components | 10 | 10 | 10 | 10 |
| seed | 7 | 7 | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | 0.0 | 0.0 | - |
| whiten | false | false | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 223.9 | 223.9..223.9 | 1 | - | - | - | transform_max_rel_err_own_fp64=3.75e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 202.9 | 202.9..202.9 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=62156.501953, transform_max_rel_err_own_fp64=3.781e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 77.9 | 77.9..77.9 | 1 | 2.874 | 2.605 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=87980.788086, transform_max_rel_err_own_fp64=3.839e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, ours-fast: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: transform(Xq)

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 126.2 | 126.2..126.2 | 1 | - | - | - | 766.5 | - | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 82.1 | 82.1..82.1 | 1 | - | - | - | 765.5 | - | explained_variance_ratio_sum=0.999997 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 142.6 | 142.6..142.6 | 1 | 0.885 | 0.576 | - | 340.8 | - | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::_linalg_eigh.eigenvalues' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on http) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows: big block, raw. Timed: fit.

mismatch: svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'

mismatch: torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it draws nothing)

mismatch: random_state is read by none of the covariance solvers; set to 7 on every arm

config: cuML benchmark (RAPIDS), PCA (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| n_components | 10 | 10 | 10 | 10 |
| seed | 7 | 7 | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | 0.0 | 0.0 | - |
| whiten | false | false | false | false |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 500000 | 56.1 | 56.1..56.1 | 1 | - | - | - | transform_max_rel_err_own_fp64=1.085e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 49.0 | 49.0..49.0 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.003017, transform_max_rel_err_own_fp64=1.186e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 17.1 | 17.1..17.1 | 1 | 3.275 | 2.860 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.719910, transform_max_rel_err_own_fp64=1.161e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, ours-fast: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: transform(Xq)

### svc / istella (rows full, shape 10000x220)

race: done, driver rc 0, log `logs/classical.svc.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 189.5 | 189.5..189.5 | 1 | - | - | - | 168.7 | - | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 525.7 | 525.7..525.7 | 1 | - | - | - | 205.2 | - | accuracy=0.922200, n_support=2401 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1320.6 | 1320.6..1320.6 | 1 | 0.143 | 0.398 | - | 277.6 | - | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| seed | null | null | 7 |
| tol | 0.001 | 0.001 | 0.001 |

accepted difference: ours seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: ours-fast seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 10000 | 90.0 | 90.0..90.0 | 1 | - | - | - | accuracy_eval=0.922200 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 10000 | 181.6 | 181.6..181.6 | 1 | - | - | - | accuracy_eval=0.922200, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 10000 | 2095.6 | 2095.6..2095.6 | 1 | 0.043 | 0.087 | - | accuracy_eval=0.922200, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn SVC.predict(Xq), host rows in, host result out

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc 0, log `logs/classical.svc.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1133.0 | 1133.0..1133.0 | 1 | - | - | - | 122.4 | - | accuracy=0.767500, n_support=5527 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 648.7 | 648.7..648.7 | 1 | - | - | - | 153.5 | - | accuracy=0.767500, n_support=5533 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3106.2 | 3106.2..3106.2 | 1 | 0.365 | 0.209 | - | 431.1 | - | accuracy=0.767500, n_support=5672 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows: svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

mismatch: seed: scikit-learn and cuML random_state=7 (read only with probability=True); ours refuses random_state without probability=True, so ours stays None

mismatch: cache_size=2000 on every arm; ours honors it only as the prediction buffer (DEVIATION 871), so its training is unaffected

mismatch: shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only

config: cuML benchmark (RAPIDS), SVC-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| seed | null | null | 7 |
| tol | 0.001 | 0.001 | 0.001 |

accepted difference: ours seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: ours-fast seed: mojolearn SVC refuses random_state without probability=True (the fit draws nothing); scikit-learn and cuML get 7

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | 10000 | 78.4 | 78.4..78.4 | 1 | - | - | - | accuracy_eval=0.767500 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 10000 | 35.3 | 35.3..35.3 | 1 | - | - | - | accuracy_eval=0.767500, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 10000 | 1565.4 | 1565.4..1565.4 | 1 | 0.050 | 0.023 | - | accuracy_eval=0.767500, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn SVC.predict(Xq), host rows in, host result out

## Classical, wave 2

### agglomerative / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.agglomerative.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1395.6 | 1395.6..1395.6 | 1 | - | - | - | 1386.8 | - | n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 416.3 | 416.3..416.3 | 1 | - | - | - | 1384.8 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5994.2 | 5994.2..5994.2 | 1 | 0.233 | 0.069 | - | 1089.7 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

config: cuML benchmark (RAPIDS), AgglomerativeClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| linkage | "single" | "single" | "single" |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 | 8 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### agglomerative / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.agglomerative.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 121.5 | 121.5..121.5 | 1 | - | - | - | 493.3 | - | n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 144.6 | 144.6..144.6 | 1 | - | - | - | 108.5 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 317.2 | 317.2..317.2 | 1 | 0.383 | 0.456 | - | 188.3 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

config: cuML benchmark (RAPIDS), AgglomerativeClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| linkage | "single" | "single" | "single" |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 | 8 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### arima / synthetic (rows full, shape Yfit 64x2000; Yhold 64x100)

race: done, driver rc 0, log `logs/classical2.arima.synthetic.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 131.7 | 131.7..131.7 | 1 | - | - | - | 87.3 | - | forecast_rmse=1.515518, insample_rmse=0.999342, mean_aic=5680.976967, mean_llf=-2836.488483 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 160.6 | 160.6..160.6 | 1 | - | - | - | 85.6 | - | forecast_rmse=1.515540, insample_rmse=0.999341, mean_aic=5680.971687, mean_llf=-2836.485844 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 638.2 | 638.2..638.2 | 1 | 0.206 | 0.252 | - | 139.5 | - | forecast_rmse=1.515423, insample_rmse=0.999338, mean_aic=5680.957160, mean_llf=-2836.478580 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: order=(1,0,1), seasonal_order=(0,0,0,0), trend='c' (cuML fit_intercept=True), maxiter=1000, maximum likelihood. Rows: 64 synthetic ARMA(1,1) series, 2000 fit points, 100 held out. Timed: fit of every series.

mismatch: ours and cuML fit the whole batch in one call; statsmodels fits one series per call (the state-space model, L-BFGS), spread over every core with joblib

mismatch: statsmodels enforce_stationarity and enforce_invertibility at its default (True); ours and cuML have no such parameter

mismatch: seed: no arm has a seed argument (maximum likelihood)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | statsmodels (declared) |
| max_iter | 1000 | 1000 | 1000 |
| order | [1, 0, 1] | [1, 0, 1] | [1, 0, 1] |
| seasonal_order | [0, 0, 0, 0] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| trend | "c" | "c" | "c" |

### elasticnet / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3836.1 | 3836.1..3836.1 | 1 | - | - | - | 2710.6 | - | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 484.5 | 484.5..484.5 | 1 | - | - | - | 3522.9 | - | finite=True, r2=0.261554, rmse=0.717827 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2116.6 | 2116.6..2116.6 | 1 | 1.812 | 0.229 | - | 2745.7 | - | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| l1_ratio | 0.5 | 0.5 | 0.5 |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | false | false | false |
| seed | null | null | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| solver | "cd" | "cd" | - |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

accepted difference: ours-fast seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

### elasticnet / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.elasticnet.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 92.9 | 92.9..92.9 | 1 | - | - | - | 183.4 | - | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 36.0 | 36.0..36.0 | 1 | - | - | - | 211.0 | - | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 53.2 | 53.2..53.2 | 1 | 1.745 | 0.676 | - | 183.8 | - | finite=True, r2=0.907378, rmse=4.847223 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), ElasticNet (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| l1_ratio | 0.5 | 0.5 | 0.5 |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | false | false | false |
| seed | null | null | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| solver | "cd" | "cd" | - |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

accepted difference: ours-fast seed: mojolearn ElasticNet refuses random_state: it selects nothing with selection='cyclic'

### ets / synthetic (rows full, shape Yfit 64x1440; Yhold 64x48)

race: done, driver rc 0, log `logs/classical2.ets.synthetic.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 537.7 | 537.7..537.7 | 1 | - | - | - | 64.9 | - | forecast_rmse=0.984392, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 355.8 | 355.8..355.8 | 1 | - | - | - | 65.3 | - | forecast_rmse=0.984473, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 1878.8 | 1878.8..1878.8 | 1 | 0.286 | 0.189 | - | 138.0 | - | forecast_rmse=0.984418, insample_rmse=0.991812 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: trend additive, seasonal additive, seasonal_periods=24, initialization_method='estimated'; ours and cuML start_periods=2, eps=2.24e-3; statsmodels damped_trend=False, use_boxcox=False. Rows: 64 synthetic hourly series, period 24, 1440 fit points, 48 held out. Timed: construct + fit of every series.

mismatch: initialization: ours 'estimated' (its default, statsmodels' definition), statsmodels 'estimated'; cuML has only its heuristic start (start_periods=2), so its row fits the older initialization

mismatch: cuML returns no in-sample predictions; that quality cell is empty

mismatch: trend: ours and cuML are additive-trend with no parameter; statsmodels trend='additive'. eps is ours' and cuML's only; statsmodels fit() uses its own optimizer

mismatch: seed: no arm has a seed argument

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsmodels (declared) |
| damped_trend | - | - | false |
| eps | 0.00224 | 0.00224 | - |
| initialization_method | "estimated" | "estimated" | "estimated" |
| seasonal | "additive" | "additive" | "additive" |
| seasonal_periods | 24 | 24 | 24 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| start_periods | 2 | 2 | - |
| trend | - | - | "additive" |

### gmm / istella (rows full, shape X 100000x200; Xq 20000x200; _dropped_constant_columns 20)

race: done, driver rc 0, log `logs/classical2.gmm.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17177.9 | 17177.9..17177.9 | 1 | - | - | - | 2171.3 | - | bic=-3.851e+07, mean_log_likelihood=200.794403, n_iter=24 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 27076.0 | 27076.0..27076.0 | 1 | - | - | - | 1670.0 | - | bic=-3.851e+07, mean_log_likelihood=200.794408, n_iter=24 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14760.5 | 14760.5..14760.5 | 1 | 1.164 | 1.834 | - | 1185.2 | - | bic=-3.901e+07, mean_log_likelihood=200.776340, n_iter=30 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| covariance_type | "full" | "full" | "full" |
| init_params | "kmeans" | "kmeans" | "kmeans" |
| max_iter | 100 | 100 | 100 |
| n_components | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| reg_covar | 0.003 | 0.003 | 0.003 |
| seed | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

### gmm / taxi (rows full, shape X 100000x11; Xq 20000x11)

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 945.4 | 945.4..945.4 | 1 | - | - | - | 236.4 | - | bic=-3.67e+06, mean_log_likelihood=12.861940, n_iter=32 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"GaussianMixture: fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to d) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, ours-fast, sklearn-cpu: host not sampled; GPU not sampled

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and 3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows: 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| covariance_type | "full" | "full" | "full" |
| init_params | "kmeans" | "kmeans" | "kmeans" |
| max_iter | 100 | 100 | 100 |
| n_components | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| reg_covar | 1e-06 | 1e-06 | 1e-06 |
| seed | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

### gpc / istella (rows full, shape X 3000x220; Xq 3000x220; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpc.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1560.6 | 1560.6..1560.6 | 1 | - | - | - | 1467.0 | - | accuracy=0.901333, logloss=0.232590, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 811.3 | 811.3..811.3 | 1 | - | - | - | 1490.4 | - | accuracy=0.901333, logloss=0.232594, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2007.0 | 2007.0..2007.0 | 1 | 0.778 | 0.404 | - | 1082.0 | - | accuracy=0.901333, logloss=0.232597, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)), optimizer=None, max_iter_predict=100, n_restarts_optimizer=0. Rows: 3000 fit and 3000 held-out stride rows of the cls block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: seed: ours refuses random_state (optimizer=None draws nothing); scikit-learn random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| kernel | "(ConstantKernel(1.0) * RBF([14.832396974191326]))" | "(ConstantKernel(1.0) * RBF([14.832396974191326]))" | "1**2 * RBF(length_scale=14.8)" |
| max_iter_predict | 100 | 100 | 100 |
| n_restarts_optimizer | 0 | 0 | 0 |
| seed | null | null | 7 |

accepted difference: ours seed: mojolearn GaussianProcessClassifier refuses random_state (optimizer=None draws nothing); scikit-learn gets 7

accepted difference: ours-fast seed: mojolearn GaussianProcessClassifier refuses random_state (optimizer=None draws nothing); scikit-learn gets 7

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### gpc / taxi (rows full, shape X 3000x11; Xq 3000x11; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpc.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1081.7 | 1081.7..1081.7 | 1 | - | - | - | 595.7 | - | accuracy=0.761000, logloss=0.541286, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 709.7 | 709.7..709.7 | 1 | - | - | - | 586.3 | - | accuracy=0.761000, logloss=0.541358, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1240.7 | 1240.7..1240.7 | 1 | 0.872 | 0.572 | - | 197.5 | - | accuracy=0.761000, logloss=0.541358, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)), optimizer=None, max_iter_predict=100, n_restarts_optimizer=0. Rows: 3000 fit and 3000 held-out stride rows of the cls block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: seed: ours refuses random_state (optimizer=None draws nothing); scikit-learn random_state=7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| kernel | "(ConstantKernel(1.0) * RBF([3.3166247903554]))" | "(ConstantKernel(1.0) * RBF([3.3166247903554]))" | "1**2 * RBF(length_scale=3.32)" |
| max_iter_predict | 100 | 100 | 100 |
| n_restarts_optimizer | 0 | 0 | 0 |
| seed | null | null | 7 |

accepted difference: ours seed: mojolearn GaussianProcessClassifier refuses random_state (optimizer=None draws nothing); scikit-learn gets 7

accepted difference: ours-fast seed: mojolearn GaussianProcessClassifier refuses random_state (optimizer=None draws nothing); scikit-learn gets 7

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### gpr / istella (rows full, shape X 3000x220; Xq 3000x220; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpr.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 453.5 | 453.5..453.5 | 1 | - | - | - | 1448.4 | - | finite=True, mean_log_predictive_density=-9.285754, r2=0.235346, rmse=0.760439 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 378.6 | 378.6..378.6 | 1 | - | - | - | 1447.3 | - | finite=True, mean_log_predictive_density=-9.285405, r2=0.235383, rmse=0.760421 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 643.2 | 643.2..643.2 | 1 | 0.705 | 0.589 | - | 1075.8 | - | finite=True, mean_log_predictive_density=-9.287148, r2=0.235368, rmse=0.760428 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)) + WhiteKernel(noise_level=0.01), alpha=2**-20 (the one ridge IDENTICAL accepts beside 0; the noise lives in the WhiteKernel so the float32 factor of K exists on taxi's near-duplicate rows), optimizer=None, normalize_y=False, n_restarts_optimizer=0, random_state=7. Rows: 3000 fit and 3000 held-out stride rows of the reg block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: kernel: the same kernel built from each library's own classes

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 9.5367431640625e-07 | 9.5367431640625e-07 | 9.5367431640625e-07 |
| kernel | "((ConstantKernel(1.0) * RBF([14.832396974191326])) + WhiteKernel(0.01))" | "((ConstantKernel(1.0) * RBF([14.832396974191326])) + WhiteKernel(0.01))" | "1**2 * RBF(length_scale=14.8) + WhiteKernel(noise_level=0.01)" |
| n_restarts_optimizer | 0 | 0 | 0 |
| seed | 7 | 7 | 7 |

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### gpr / taxi (rows full, shape X 3000x11; Xq 3000x11; y 3000; yq 3000)

race: done, driver rc 0, log `logs/classical2.gpr.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 350.3 | 350.3..350.3 | 1 | - | - | - | 562.3 | - | finite=True, mean_log_predictive_density=-311.458394, r2=0.889630, rmse=5.041639 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 261.6 | 261.6..261.6 | 1 | - | - | - | 556.1 | - | finite=True, mean_log_predictive_density=-311.471466, r2=0.889629, rmse=5.041669 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 230.9 | 230.9..230.9 | 1 | 1.517 | 1.133 | - | 186.0 | - | finite=True, mean_log_predictive_density=-311.539594, r2=0.889629, rmse=5.041653 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)) + WhiteKernel(noise_level=0.01), alpha=2**-20 (the one ridge IDENTICAL accepts beside 0; the noise lives in the WhiteKernel so the float32 factor of K exists on taxi's near-duplicate rows), optimizer=None, normalize_y=False, n_restarts_optimizer=0, random_state=7. Rows: 3000 fit and 3000 held-out stride rows of the reg block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: kernel: the same kernel built from each library's own classes

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 9.5367431640625e-07 | 9.5367431640625e-07 | 9.5367431640625e-07 |
| kernel | "((ConstantKernel(1.0) * RBF([3.3166247903554])) + WhiteKernel(0.01))" | "((ConstantKernel(1.0) * RBF([3.3166247903554])) + WhiteKernel(0.01))" | "1**2 * RBF(length_scale=3.32) + WhiteKernel(noise_level=0.01)" |
| n_restarts_optimizer | 0 | 0 | 0 |
| seed | 7 | 7 | 7 |

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### ivf / istella (rows full, shape index 400000x220; queries 4000x220)

race: done, driver rc 0, log `logs/classical2.ivf.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20884.5 | 20884.5..20884.5 | 1 | - | - | - | 1695.6 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19500.3 | 19500.3..19500.3 | 1 | - | - | - | 2212.2 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 14259.0 | 14259.0..14259.0 | 1 | 1.465 | 1.368 | - | 1260.5 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (get_params) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

### ivf / taxi (rows full, shape index 400000x11; queries 4000x11)

race: done, driver rc 0, log `logs/classical2.ivf.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1828.4 | 1828.4..1828.4 | 1 | - | - | - | 443.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1003.9 | 1003.9..1003.9 | 1 | - | - | - | 466.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 7542.8 | 7542.8..7542.8 | 1 | 0.242 | 0.133 | - | 138.6 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows: the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (get_params) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 | 10 |
| nlist | 1024 | 1024 | 1024 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

### kernel-ridge / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1890.8 | 1890.8..1890.8 | 1 | - | - | - | 1817.3 | - | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 764.8 | 764.8..764.8 | 1 | - | - | - | 1818.5 | - | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1857.4 | 1857.4..1857.4 | 1 | 1.018 | 0.412 | - | 1854.2 | - | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 | 1.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### kernel-ridge / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1885.9 | 1885.9..1885.9 | 1 | - | - | - | 901.4 | - | finite=True, r2=0.726543, rmse=8.330373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 460.4 | 460.4..460.4 | 1 | - | - | - | 897.7 | - | finite=True, r2=0.726543, rmse=8.330374 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1850.3 | 1850.3..1850.3 | 1 | 1.019 | 0.249 | - | 975.3 | - | finite=True, r2=0.726542, rmse=8.330380 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

config: cuML benchmark (RAPIDS), KernelRidge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 | 1.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### knn-clf / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 594.0 | 594.0..594.0 | 1 | - | - | - | 1689.1 | - | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3282.0 | 3282.0..3282.0 | 1 | - | - | - | 1623.9 | - | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 665.9 | 665.9..665.9 | 1 | 0.892 | 4.929 | - | 1257.3 | - | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" | "uniform" |

### knn-clf / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-clf.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 132.0 | 132.0..132.0 | 1 | - | - | - | 307.9 | - | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 104.7 | 104.7..104.7 | 1 | - | - | - | 226.8 | - | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 312.2 | 312.2..312.2 | 1 | 0.423 | 0.335 | - | 206.8 | - | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" | "uniform" |

### knn-reg / istella (rows full, shape X 200000x220; Xq 4000x220; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 572.7 | 572.7..572.7 | 1 | - | - | - | 1688.7 | - | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3260.6 | 3260.6..3260.6 | 1 | - | - | - | 1619.4 | - | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 656.2 | 656.2..656.2 | 1 | 0.873 | 4.969 | - | 1261.5 | - | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" | "uniform" |

### knn-reg / taxi (rows full, shape X 200000x11; Xq 4000x11; y 200000; yq 4000)

race: done, driver rc 0, log `logs/classical2.knn-reg.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 104.6 | 104.6..104.6 | 1 | - | - | - | 299.6 | - | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 84.7 | 84.7..84.7 | 1 | - | - | - | 226.1 | - | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 283.9 | 283.9..283.9 | 1 | 0.368 | 0.298 | - | 207.0 | - | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows: 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

config: cuML benchmark (RAPIDS), KNeighborsRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" | "uniform" |

### lasso / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11129.5 | 11129.5..11129.5 | 1 | - | - | - | 2711.2 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 488.4 | 488.4..488.4 | 1 | - | - | - | 3522.3 | - | finite=True, r2=0.310472, rmse=0.693643 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4656.9 | 4656.9..4656.9 | 1 | 2.390 | 0.105 | - | 2744.7 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 | 0.01 |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | false | false | false |
| seed | null | null | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| solver | "cd" | "cd" | - |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

accepted difference: ours-fast seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

### lasso / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.lasso.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 105.5 | 105.5..105.5 | 1 | - | - | - | 184.9 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 49.1 | 49.1..49.1 | 1 | - | - | - | 212.0 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 63.0 | 63.0..63.0 | 1 | 1.676 | 0.780 | - | 187.6 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

config: cuML benchmark (RAPIDS), Lasso (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 | 0.01 |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | false | false | false |
| seed | null | null | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| solver | "cd" | "cd" | - |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

accepted difference: ours-fast seed: mojolearn Lasso refuses random_state: it selects nothing with selection='cyclic'

### linearsvc / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4570.2 | 4570.2..4570.2 | 1 | - | - | - | 1875.4 | - | accuracy=0.923480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3419.1 | 3419.1..3419.1 | 1 | - | - | - | 1875.8 | - | accuracy=0.923230 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 636636.9 | 636636.9..636636.9 | 1 | 0.007 | 0.005 | - | 5920.5 | - | accuracy=0.923540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| loss | "squared_hinge" | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 | 1000 |
| penalized_intercept | false | false | - |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

### linearsvc / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvc.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 152.4 | 152.4..152.4 | 1 | - | - | - | 216.3 | - | accuracy=0.763330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 145.2 | 145.2..145.2 | 1 | - | - | - | 198.7 | - | accuracy=0.763330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 609.2 | 609.2..609.2 | 1 | 0.250 | 0.238 | - | 269.9 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

config: cuML benchmark (RAPIDS), LinearSVC (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| loss | "squared_hinge" | "squared_hinge" | "squared_hinge" |
| max_iter | 1000 | 1000 | 1000 |
| penalized_intercept | false | false | - |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

### linearsvr / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6522.6 | 6522.6..6522.6 | 1 | - | - | - | 1840.4 | - | finite=True, r2=-0.106754, rmse=0.878792 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4519.3 | 4519.3..4519.3 | 1 | - | - | - | 1838.3 | - | finite=True, r2=-0.106754, rmse=0.878792 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 740853.5 | 740853.5..740853.5 | 1 | 0.009 | 0.006 | - | 5950.5 | - | finite=True, r2=-0.025729, rmse=0.846012 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 | 0.0 |
| fit_intercept | true | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 | 1000 |
| penalized_intercept | false | false | - |
| penalty | "l2" | "l2" | - |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

### linearsvr / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.linearsvr.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 249.5 | 249.5..249.5 | 1 | - | - | - | 163.6 | - | finite=True, r2=0.899813, rmse=5.041302 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 464.7 | 464.7..464.7 | 1 | - | - | - | 161.6 | - | finite=True, r2=0.899814, rmse=5.041262 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 90466.9 | 90466.9..90466.9 | 1 | 0.003 | 0.005 | - | 240.6 | - | finite=True, r2=0.899803, rmse=5.041552 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

config: cuML benchmark (RAPIDS), LinearSVR (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| epsilon | 0.0 | 0.0 | 0.0 |
| fit_intercept | true | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 1000 | 1000 | 1000 |
| penalized_intercept | false | false | - |
| penalty | "l2" | "l2" | - |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

### logreg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26847.3 | 26847.3..26847.3 | 1 | - | - | - | 1849.3 | - | accuracy=0.924540, logloss=0.181245, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19865.2 | 19865.2..19865.2 | 1 | - | - | - | 1849.4 | - | accuracy=0.924570, logloss=0.181257, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 50824.9 | 50824.9..50824.9 | 1 | 0.528 | 0.391 | - | 2747.6 | - | accuracy=0.924470, logloss=0.181337, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| l1_ratio | null | null | null |
| max_iter | 1000 | 1000 | 1000 |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| solver | "qn" | "qn" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: ours-fast l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu solver: L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'

### logreg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.logreg.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 295.9 | 295.9..295.9 | 1 | - | - | - | 165.7 | - | accuracy=0.763340, logloss=0.538984, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 367.7 | 367.7..367.7 | 1 | - | - | - | 170.3 | - | accuracy=0.763350, logloss=0.538986, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 374.5 | 374.5..374.5 | 1 | 0.790 | 0.982 | - | 194.3 | - | accuracy=0.763320, logloss=0.538980, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

config: cuML benchmark (RAPIDS), LogisticRegression (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| l1_ratio | null | null | null |
| max_iter | 1000 | 1000 | 1000 |
| penalty | "l2" | "l2" | "l2" |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| solver | "qn" | "qn" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: ours-fast l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu solver: L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'

### nystroem / istella (rows full, shape X 100000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.nystroem.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 323.8 | 323.8..323.8 | 1 | - | - | - | 1342.8 | - | kernel_rel_error=0.038958 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the landmark rows are each library's own random draw from seed 7

mismatch: degree=3, coef0=1.0 on every arm; the rbf kernel reads neither

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 1.0 | 1.0 | 1.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### nystroem / taxi (rows full, shape X 100000x11; Xcheck 1000x11)

race: done, driver rc 0, log `logs/classical2.nystroem.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 260.4 | 260.4..260.4 | 1 | - | - | - | 393.6 | - | kernel_rel_error=0.044370 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the landmark rows are each library's own random draw from seed 7

mismatch: degree=3, coef0=1.0 on every arm; the rbf kernel reads neither

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 1.0 | 1.0 | 1.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### rbf-sampler / istella (rows full, shape X 100000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.rbf-sampler.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 90.8 | 90.8..90.8 | 1 | - | - | - | 1597.5 | - | kernel_rel_error=0.141980 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 159.9 | 159.9..159.9 | 1 | - | - | - | 1594.5 | - | kernel_rel_error=0.141980 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 64.3 | 64.3..64.3 | 1 | 1.411 | 2.485 | - | 1344.6 | - | kernel_rel_error=0.137405 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the random Fourier features are each library's own draw from seed 7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### rbf-sampler / taxi (rows full, shape X 100000x11; Xcheck 1000x11)

race: done, driver rc 0, log `logs/classical2.rbf-sampler.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.9 | 54.9..54.9 | 1 | - | - | - | 478.6 | - | kernel_rel_error=0.108549 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 60.0 | 60.0..60.0 | 1 | - | - | - | 473.5 | - | kernel_rel_error=0.108549 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 51.7 | 51.7..51.7 | 1 | 1.063 | 1.162 | - | 386.7 | - | kernel_rel_error=0.083775 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: gamma=1/d, n_components=256, random_state=7. Rows: 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the random Fourier features are each library's own draw from seed 7

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.ridge.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2875.0 | 2875.0..2875.0 | 1 | - | - | - | 5192.5 | - | finite=True, r2=0.328682, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2168.6 | 2168.6..2168.6 | 1 | - | - | - | 5186.2 | - | finite=True, r2=0.320451, rmse=0.688606 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8827.1 | 8827.1..8827.1 | 1 | 0.326 | 0.246 | - | 8619.6 | - | finite=True, r2=0.328676, rmse=0.684426 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| fit_intercept | true | true | true |
| max_iter | - | - | null |
| normalize | false | false | - |
| positive | - | - | false |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| solver | "eig" | "eig" | "cholesky" |
| tol | - | - | 0.0001 |

accepted difference: sklearn-cpu solver: ours and cuML 'eig' (eigendecomposition of the normal equations); scikit-learn has no 'eig' and runs 'cholesky' on the same normal equations

### ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/classical2.ridge.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 53.6 | 53.6..53.6 | 1 | - | - | - | 281.6 | - | finite=True, r2=0.908983, rmse=4.805042 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 51.9 | 51.9..51.9 | 1 | - | - | - | 280.0 | - | finite=True, r2=0.908983, rmse=4.805048 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 44.6 | 44.6..44.6 | 1 | 1.202 | 1.164 | - | 185.9 | - | finite=True, r2=0.908988, rmse=4.804916 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows: 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

config: cuML benchmark (RAPIDS), Ridge (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| fit_intercept | true | true | true |
| max_iter | - | - | null |
| normalize | false | false | - |
| positive | - | - | false |
| seed | "none (deterministic)" | "none (deterministic)" | 7 |
| solver | "eig" | "eig" | "cholesky" |
| tol | - | - | 0.0001 |

accepted difference: sklearn-cpu solver: ours and cuML 'eig' (eigendecomposition of the normal equations); scikit-learn has no 'eig' and runs 'cholesky' on the same normal equations

### spectral-embedding / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 666.5 | 666.5..666.5 | 1 | - | - | - | 225.4 | - | trustworthiness_k15=0.799378 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1687.1 | 1687.1..1687.1 | 1 | - | - | - | 188.0 | - | trustworthiness_k15=0.823640 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10499.0 | 10499.0..10499.0 | 1 | 0.063 | 0.161 | - | 189.6 | - | trustworthiness_k15=0.812688 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows: the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| gamma | null | null | null |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral-embedding / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 260.8 | 260.8..260.8 | 1 | - | - | - | 119.8 | - | trustworthiness_k15=0.884889 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 135.8 | 135.8..135.8 | 1 | - | - | - | 99.5 | - | trustworthiness_k15=0.895236 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2464.5 | 2464.5..2464.5 | 1 | 0.106 | 0.055 | - | 151.2 | - | trustworthiness_k15=0.898011 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows: the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| gamma | null | null | null |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.spectral.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 379.4 | 379.4..379.4 | 1 | - | - | - | 1088.6 | - | n_clusters=8, silhouette=0.147668 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 528.2 | 528.2..528.2 | 1 | - | - | - | 1060.4 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.147668 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1849.4 | 1849.4..1849.4 | 1 | 0.205 | 0.286 | - | 1116.2 | - | ari_vs_ours=0.999826, n_clusters=8, silhouette=0.147699 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, random_state=42 (the cuML benchmark's SpectralClustering), assign_labels='kmeans', n_components=8. Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

config: cuML benchmark (RAPIDS), SpectralClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| assign_labels | "kmeans" | "kmeans" | "kmeans" |
| coef0 | - | - | 1 |
| degree | - | - | 3 |
| gamma | null | null | 1.0 |
| n_clusters | 8 | 8 | 8 |
| n_components | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| n_neighbors | 10 | 10 | 10 |
| seed | 42 | 42 | 42 |

accepted difference: ours-fast gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral / taxi (rows full, shape X 10000x11)

race: done, driver rc 0, log `logs/classical2.spectral.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 437.8 | 437.8..437.8 | 1 | - | - | - | 156.6 | - | n_clusters=8, silhouette=0.039910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 163.9 | 163.9..163.9 | 1 | - | - | - | 155.3 | - | ari_vs_ours=0.997088, n_clusters=8, silhouette=0.039888 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 997.5 | 997.5..997.5 | 1 | 0.439 | 0.164 | - | 199.1 | - | ari_vs_ours=0.582313, n_clusters=8, silhouette=0.089894 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, random_state=42 (the cuML benchmark's SpectralClustering), assign_labels='kmeans', n_components=8. Rows: 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

config: cuML benchmark (RAPIDS), SpectralClustering (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 42): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| affinity | "nearest_neighbors" | "nearest_neighbors" | "nearest_neighbors" |
| assign_labels | "kmeans" | "kmeans" | "kmeans" |
| coef0 | - | - | 1 |
| degree | - | - | 3 |
| gamma | null | null | 1.0 |
| n_clusters | 8 | 8 | 8 |
| n_components | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| n_neighbors | 10 | 10 | 10 |
| seed | 42 | 42 | 42 |

accepted difference: ours-fast gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### svr / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.svr.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 219.0 | 219.0..219.0 | 1 | - | - | - | 1099.2 | - | finite=True, r2=0.318258, rmse=0.680816 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 527.1 | 527.1..527.1 | 1 | - | - | - | 1140.2 | - | finite=True, r2=0.318257, rmse=0.680816 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1772.9 | 1772.9..1772.9 | 1 | 0.124 | 0.297 | - | 1136.5 | - | finite=True, r2=0.318248, rmse=0.680821 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| epsilon | 0.1 | 0.1 | 0.1 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

### svr / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/classical2.svr.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 180.0 | 180.0..180.0 | 1 | - | - | - | 164.6 | - | finite=True, r2=0.767551, rmse=7.680395 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 116.6 | 116.6..116.6 | 1 | - | - | - | 200.5 | - | finite=True, r2=0.767550, rmse=7.680415 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1488.0 | 1488.0..1488.0 | 1 | 0.121 | 0.078 | - | 507.6 | - | finite=True, r2=0.767550, rmse=7.680414 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows: 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

config: cuML benchmark (RAPIDS), SVR-RBF (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| epsilon | 0.1 | 0.1 | 0.1 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

### tsvd / istella (rows full, shape X 1000000x220)

race: done, driver rc 0, log `logs/classical2.tsvd.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1886.3 | 1886.3..1886.3 | 1 | - | - | - | 3417.4 | - | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.002554 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 603.5 | 603.5..603.5 | 1 | - | - | - | 3416.9 | - | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.002554 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1970.1 | 1970.1..1970.1 | 1 | 0.957 | 0.306 | - | 1825.5 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.000122 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "covariance_eigh" | "covariance_eigh" | "arpack" |
| n_components | 10 | 10 | 10 |
| n_iter | 5 | 5 | 5 |
| seed | 7 | 7 | 7 |
| tol | 0.0 | 0.0 | 0.0 |

accepted difference: sklearn-cpu algorithm: scikit-learn TruncatedSVD has no 'covariance_eigh'; it runs 'arpack' at tol=0

### tsvd / taxi (rows full, shape X 1000000x11)

race: done, driver rc 0, log `logs/classical2.tsvd.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 50.0 | 50.0..50.0 | 1 | - | - | - | 306.6 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 59.9 | 59.9..59.9 | 1 | - | - | - | 305.0 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 199.6 | 199.6..199.6 | 1 | 0.251 | 0.300 | - | 184.3 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows: 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

config: cuML benchmark (RAPIDS), tSVD (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "covariance_eigh" | "covariance_eigh" | "arpack" |
| n_components | 10 | 10 | 10 |
| n_iter | 5 | 5 | 5 |
| seed | 7 | 7 | 7 |
| tol | 0.0 | 0.0 | 0.0 |

accepted difference: sklearn-cpu algorithm: scikit-learn TruncatedSVD has no 'covariance_eigh'; it runs 'arpack' at tol=0

### umap / istella (rows full, shape X 20000x220)

race: done, driver rc 0, log `logs/classical2.umap.istella.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1631.1 | 1631.1..1631.1 | 1 | - | - | - | 259.6 | - | trustworthiness_k15=0.979906 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2461.1 | 2461.1..2461.1 | 1 | - | - | - | 219.1 | - | trustworthiness_k15=0.981706 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 10997.8 | 10997.8..10997.8 | 1 | 0.148 | 0.224 | - | 487.3 | - | trustworthiness_k15=0.978822 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 2837.8 | 2837.8..2837.8 | 1 | 0.575 | 0.867 | - | 497.4 | - | trustworthiness_k15=0.977512 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows: 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

config: cuML benchmark (RAPIDS), UMAP-Unsupervised (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | umap-learn-cpu | umap-learn-cpu-unseeded |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | umap-learn (get_params) | umap-learn (get_params) |
| init | "spectral" | "spectral" | "spectral" | "spectral" |
| learning_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| local_connectivity | 1.0 | 1.0 | 1.0 | 1.0 |
| metric | "euclidean" | "euclidean" | "euclidean" | "euclidean" |
| min_dist | 0.1 | 0.1 | 0.1 | 0.1 |
| n_components | 2 | 2 | 2 | 2 |
| n_epochs | 500 | 500 | 500 | 500 |
| n_neighbors | 5 | 5 | 5 | 5 |
| negative_sample_rate | 5 | 5 | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 | 1.0 | 1.0 |
| seed | 7 | 7 | 7 | null |
| set_op_mix_ratio | 1.0 | 1.0 | 1.0 | 1.0 |
| spread | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: umap-learn-cpu-unseeded seed: raced unseeded on purpose: seeded umap-learn runs one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)

### umap / taxi (rows full, shape X 20000x11)

race: done, driver rc 0, log `logs/classical2.umap.taxi.rows-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 568.1 | 568.1..568.1 | 1 | - | - | - | 129.0 | - | trustworthiness_k15=0.990480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 576.6 | 576.6..576.6 | 1 | - | - | - | 115.8 | - | trustworthiness_k15=0.991729 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 9646.3 | 9646.3..9646.3 | 1 | 0.059 | 0.060 | - | 453.2 | - | trustworthiness_k15=0.989525 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 2603.3 | 2603.3..2603.3 | 1 | 0.218 | 0.221 | - | 457.2 | - | trustworthiness_k15=0.990395 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows: 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

config: cuML benchmark (RAPIDS), UMAP-Unsupervised (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | umap-learn-cpu | umap-learn-cpu-unseeded |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | umap-learn (get_params) | umap-learn (get_params) |
| init | "spectral" | "spectral" | "spectral" | "spectral" |
| learning_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| local_connectivity | 1.0 | 1.0 | 1.0 | 1.0 |
| metric | "euclidean" | "euclidean" | "euclidean" | "euclidean" |
| min_dist | 0.1 | 0.1 | 0.1 | 0.1 |
| n_components | 2 | 2 | 2 | 2 |
| n_epochs | 500 | 500 | 500 | 500 |
| n_neighbors | 5 | 5 | 5 | 5 |
| negative_sample_rate | 5 | 5 | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 | 1.0 | 1.0 |
| seed | 7 | 7 | 7 | null |
| set_op_mix_ratio | 1.0 | 1.0 | 1.0 | 1.0 |
| spread | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: umap-learn-cpu-unseeded seed: raced unseeded on purpose: seeded umap-learn runs one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)

## Neural

### gemm-bf16 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-bf16.gaussian.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 159.9 | 159.9..159.9 | 1 | - | - | - | 627.0 | - | max_rel_err_vs_fp64=1.155e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-bf16 | torch | gpu | opponent | 50.1 | 50.1..50.1 | 1 | 3.189 | - | - | 1451.3 | 1032.4 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 51.0 | 51.0..51.0 | 1 | 3.136 | - | - | 1608.0 | 1032.4 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-eager-bf16 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### gemm-int8 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-int8.gaussian.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6425.1 | 6425.1..6425.1 | 1 | - | - | - | 234.7 | - | max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours |
|---|---|
| library (source) | mojolearn (declared) |
| seed | "none (deterministic)" |

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm.gaussian.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 160.0 | 160.0..160.0 | 1 | - | - | - | 502.3 | - | max_rel_err_vs_fp64=2.399e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 43.8 | 43.8..43.8 | 1 | 3.656 | - | - | 1374.8 | 1024.4 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.843e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 28.8 | 28.8..28.8 | 1 | 5.548 | - | - | 1534.5 | 1024.4 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.843e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 52.8 | 52.8..52.8 | 1 | 3.031 | - | - | 1386.6 | 1032.4 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 53.3 | 53.3..53.3 | 1 | 3.001 | - | - | 1551.9 | 1032.4 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-forward.bytes.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 447.4 | 447.4..447.4 | 1 | - | - | - | 4296.4 | - | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 53.2 | 53.2..53.2 | 1 | 8.407 | - | - | 1367.9 | 1048.4 | max_abs_diff_vs_ours=1.241e-06, max_rel_diff_vs_ours=1.703e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 37.5 | 37.5..37.5 | 1 | 11.927 | - | - | 1524.5 | 1048.4 | max_abs_diff_vs_ours=1.237e-06, max_rel_diff_vs_ours=1.698e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 66.8 | 66.8..66.8 | 1 | 6.695 | - | - | 1398.6 | 1080.6 | max_abs_diff_vs_ours=0.006710, max_rel_diff_vs_ours=0.009211, mean_nll=9.018647 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 37.4 | 37.4..37.4 | 1 | 11.947 | - | - | 1528.6 | 1048.6 | max_abs_diff_vs_ours=0.006710, max_rel_diff_vs_ours=0.009211, mean_nll=9.018634 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-host-train-step / bytes (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-host-train-step.bytes.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 2334.2 | 2334.2..2334.2 | 1 | - | - | - | 2178.8 | - | loss_first_step=9.017858, loss_last_step=8.367768, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 150.3 | 150.3..150.3 | 1 | 15.534 | - | - | 617.7 | - | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.017857, loss_last_abs_diff_vs_ours=1.907e-06, loss_last_step=8.367766, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 5741.8 | 5741.8..5741.8 | 1 | 0.407 | - | - | 598.0 | - | loss_first_abs_diff_vs_ours=9.155e-05, loss_first_step=9.017766, loss_last_abs_diff_vs_ours=0.001106, loss_last_step=8.368875, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-cpu-compile-fp32, torch-cpu-compile-bf16: host not sampled; GPU not sampled

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

race: done, driver rc 0, log `logs/neural.lm-infer.bytes.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 1618.5 | 1618.5..1618.5 | 1 | - | - | - | 451.0 | - | mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 41.3 | 41.3..41.3 | 1 | 39.214 | - | - | 340.5 | - | max_abs_diff_vs_ours=9.239e-07, max_rel_diff_vs_ours=1.268e-06, mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 174.5 | 174.5..174.5 | 1 | 9.274 | - | - | 352.6 | - | max_abs_diff_vs_ours=0.006131, max_rel_diff_vs_ours=0.008417, mean_nll=9.017766 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-cpu-compile-fp32, torch-cpu-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-train-step.bytes.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 680.4 | 680.4..680.4 | 1 | - | - | - | 3747.6 | - | loss_first_step=9.018733, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 216.2 | 216.2..216.2 | 1 | 3.147 | - | - | 3621.5 | 3226.6 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=9.018733, loss_last_abs_diff_vs_ours=0.000000, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 149.7 | 149.7..149.7 | 1 | 4.544 | - | - | 2660.0 | 2168.4 | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.018734, loss_last_abs_diff_vs_ours=1.907e-06, loss_last_step=8.422413, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 281.6 | 281.6..281.6 | 1 | 2.416 | - | - | 2616.5 | 2226.6 | loss_first_abs_diff_vs_ours=0.003331, loss_first_step=9.022064, loss_last_abs_diff_vs_ours=0.003633, loss_last_step=8.418777, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 175.7 | 175.7..175.7 | 1 | 3.873 | - | - | 2708.8 | 2200.6 | loss_first_abs_diff_vs_ours=0.0003309, loss_first_step=9.018402, loss_last_abs_diff_vs_ours=0.002153, loss_last_step=8.420258, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

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

race: done, driver rc 0, log `logs/neural.mamba1-forward.gaussian.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 30.1 | 30.1..30.1 | 1 | - | - | - | 140.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 106.5 | 106.5..106.5 | 1 | 0.283 | - | - | 1487.8 | 1232.4 | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.189e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 113.8 | 113.8..113.8 | 1 | 0.265 | - | - | 1456.3 | 1200.6 | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.568e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |
| ssm_d_conv | 4 | - | - |
| ssm_d_state | 16 | - | - |
| ssm_dt_rank | 24 | - | - |
| ssm_expand | 2 | - | - |

### mamba1-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-infer.gaussian.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 92.8 | 92.8..92.8 | 1 | - | - | - | 80.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 58.7 | 58.7..58.7 | 1 | 1.582 | - | - | 269.9 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.948e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 186.9 | 186.9..186.9 | 1 | 0.497 | - | - | 256.7 | - | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.57e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-forward.gaussian.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 78.0 | 78.0..78.0 | 1 | - | - | - | 210.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 1252.9 | 1252.9..1252.9 | 1 | 0.062 | - | - | 7491.2 | 7248.7 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 2765.4 | 2765.4..2765.4 | 1 | 0.028 | - | - | 7601.6 | 7248.7 | max_abs_diff_vs_ours=2.146e-06, max_rel_diff_vs_ours=7.295e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 2547.0 | 2547.0..2547.0 | 1 | 0.031 | - | - | 5988.3 | 5744.7 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 2439.0 | 2439.0..2439.0 | 1 | 0.032 | - | - | 6100.1 | 5744.7 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| ssm_chunk_size | 256 | - | - | - | - |
| ssm_d_conv | 4 | - | - | - | - |
| ssm_d_state | 128 | - | - | - | - |
| ssm_dt_limit | [0.0, Infinity] | - | - | - | - |
| ssm_expand | 2 | - | - | - | - |
| ssm_headdim | 64 | - | - | - | - |
| ssm_ngroups | 1 | - | - | - | - |

### mamba2-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-infer.gaussian.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 62.6 | 62.6..62.6 | 1 | - | - | - | 97.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 127.1 | 127.1..127.1 | 1 | 0.493 | - | - | 1014.0 | - | max_abs_diff_vs_ours=1.192e-06, max_rel_diff_vs_ours=4.233e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 126.0 | 126.0..126.0 | 1 | 0.497 | - | - | 1129.2 | - | max_abs_diff_vs_ours=1.192e-06, max_rel_diff_vs_ours=4.233e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 120.2 | 120.2..120.2 | 1 | 0.521 | - | - | 245.6 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 118.8 | 118.8..118.8 | 1 | 0.527 | - | - | 357.2 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-forward.gaussian.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 45.6 | 45.6..45.6 | 1 | - | - | - | 200.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 147.7 | 147.7..147.7 | 1 | 0.309 | - | - | 1349.5 | 1096.7 | max_abs_diff_vs_ours=7.153e-07, max_rel_diff_vs_ours=3.129e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/reduction_utils.h>\\n#include <c10/metal/utils) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 96.7 | 96.7..96.7 | 1 | 0.472 | - | - | 1351.8 | 1096.7 | max_abs_diff_vs_ours=0.002095, max_rel_diff_vs_ours=0.0009164 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/reduction_utils.h>\\n#include <c10/metal/utils) (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

memory, torch-compile-fp32, torch-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba3-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-infer.gaussian.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 70.4 | 70.4..70.4 | 1 | - | - | - | 101.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 23.7 | 23.7..23.7 | 1 | 2.967 | - | - | 248.3 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.119e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 51.3 | 51.3..51.3 | 1 | 1.372 | - | - | 242.8 | - | max_abs_diff_vs_ours=0.002148, max_rel_diff_vs_ours=0.0009547 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-cpu-compile-fp32, torch-cpu-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-infer / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-infer.gaussian.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 45.4 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 0.3 | 0.3..0.3 | 1 | 0.957 | - | - | 150.9 | - | max_abs_diff_vs_ours=0.000000, max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 0.6 | 0.6..0.6 | 1 | 0.523 | - | - | 153.0 | - | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-cpu-compile-fp32, torch-cpu-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-train-step.gaussian.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.9 | 8.9..8.9 | 1 | - | - | - | 54.7 | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 4.6 | 4.6..4.6 | 1 | 1.948 | - | - | 258.8 | 18.6 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 4.3 | 4.3..4.3 | 1 | 2.066 | - | - | 352.6 | 16.4 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.5 | 4.5..4.5 | 1 | 1.979 | - | - | 261.1 | 18.6 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 3.6 | 3.6..3.6 | 1 | 2.484 | - | - | 353.3 | 16.6 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001007, loss_last_step=1.123462, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

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

### samba-forward / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-forward.bytes.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 96.4 | 96.4..96.4 | 1 | - | - | - | 329.0 | - | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 57.8 | 57.8..57.8 | 1 | 1.667 | - | - | 440.5 | 168.7 | max_abs_diff_vs_ours=2.623e-06, max_rel_diff_vs_ours=1.474e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/error.h>\\n#include <c10/metal/reduction_utils) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 52.8 | 52.8..52.8 | 1 | 1.826 | - | - | 438.6 | 168.7 | max_abs_diff_vs_ours=0.016692, max_rel_diff_vs_ours=0.009384, mean_nll=5.636023 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/error.h>\\n#include <c10/metal/reduction_utils) (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

memory, torch-compile-fp32, torch-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### samba-infer / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-infer.bytes.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 440.0 | 440.0..440.0 | 1 | - | - | - | 208.9 | - | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 87.4 | 87.4..87.4 | 1 | 5.035 | - | - | 260.9 | - | max_abs_diff_vs_ours=2.682e-06, max_rel_diff_vs_ours=1.508e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 306.4 | 306.4..306.4 | 1 | 1.436 | - | - | 265.2 | - | max_abs_diff_vs_ours=0.018826, max_rel_diff_vs_ours=0.010584, mean_nll=5.636032 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(timeout: null) (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-cpu-compile-fp32, torch-cpu-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-train-step.bytes.shape-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2099.8 | 2099.8..2099.8 | 1 | - | - | - | 596.3 | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 153.2 | 153.2..153.2 | 1 | 13.704 | - | - | 1650.1 | 1322.8 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/error.h>\\n#include <c10/metal/reduction_utils) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 148.3 | 148.3..148.3 | 1 | 14.159 | - | - | 1632.2 | 1295.8 | loss_first_abs_diff_vs_ours=0.0001116, loss_first_step=5.636022, loss_last_abs_diff_vs_ours=0.0001273, loss_last_step=4.833807, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-bf16 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/error.h>\\n#include <c10/metal/reduction_utils) (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

memory, torch-compile-fp32, torch-compile-bf16: host not sampled; GPU not sampled

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### transformer-forward / gaussian (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-forward.gaussian.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 55.7 | 55.7..55.7 | 1 | - | - | - | 266.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 14.0 | 14.0..14.0 | 1 | 3.985 | - | - | 356.5 | 112.4 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 15.3 | 15.3..15.3 | 1 | 3.649 | - | - | 1402.1 | 1064.4 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 13.5 | 13.5..13.5 | 1 | 4.129 | - | - | 361.0 | 112.7 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 6.9 | 6.9..6.9 | 1 | 8.105 | - | - | 387.2 | 40.6 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### transformer-infer / gaussian (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024)

race: done, driver rc 0, log `logs/neural.transformer-infer.gaussian.shape-full.log`, ran on ip-172-31-40-10.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 44.2 | 44.2..44.2 | 1 | - | - | - | 116.6 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 7.6 | 7.6..7.6 | 1 | 5.820 | - | - | 235.6 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.076e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 5.5 | 5.5..5.5 | 1 | 8.029 | - | - | 343.3 | - | max_abs_diff_vs_ours=9.775e-06, max_rel_diff_vs_ours=2.207e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 18.6 | 18.6..18.6 | 1 | 2.376 | - | - | 240.4 | - | max_abs_diff_vs_ours=0.001941, max_rel_diff_vs_ours=0.0004383 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 16.3 | 16.3..16.3 | 1 | 2.707 | - | - | 344.6 | - | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0004107 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

## Algorithm expansion

### ard / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.ard.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 94167.8 | 94167.8..94167.8 | 1 | - | - | - | 1249.7 | - | finite=True, r2=-0.122912, rmse=0.885183 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 87906.2 | 87906.2..87906.2 | 1 | - | - | - | 1251.4 | - | finite=True, r2=-0.123501, rmse=0.885416 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13189.1 | 13189.1..13189.1 | 1 | 7.140 | 6.665 | - | 1161.0 | - | finite=True, r2=0.327436, rmse=0.685058 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.5 | 13.5..13.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 13.2 | 13.2..13.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.9 | 5.9..5.9 | 1 | 2.287 | 2.246 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ard / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.ard.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 86.6 | 86.6..86.6 | 1 | - | - | - | 129.1 | - | finite=True, r2=0.909193, rmse=4.799513 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 77.5 | 77.5..77.5 | 1 | - | - | - | 133.2 | - | finite=True, r2=0.909193, rmse=4.799513 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20.5 | 20.5..20.5 | 1 | 4.219 | 3.775 | - | 208.0 | - | finite=True, r2=0.909190, rmse=4.799575 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 3.719 | 3.753 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bayesian-ridge.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 249721.3 | 249721.3..249721.3 | 1 | - | - | - | 1938.8 | - | finite=True, r2=-41643.670747, rmse=170.466932 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 237514.9 | 237514.9..237514.9 | 1 | - | - | - | 1939.6 | - | finite=True, r2=-41688.617698, rmse=170.558899 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10501.3 | 10501.3..10501.3 | 1 | 23.780 | 22.618 | - | 4436.6 | - | finite=True, r2=-890.186917, rmse=24.937036 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 14.2 | 14.2..14.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.4 | 15.4..15.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.0 | 6.0..6.0 | 1 | 2.378 | 2.581 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bayesian-ridge.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 741.3 | 741.3..741.3 | 1 | - | - | - | 180.8 | - | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 654.2 | 654.2..654.2 | 1 | - | - | - | 183.7 | - | finite=True, r2=0.908981, rmse=4.805108 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 91.9 | 91.9..91.9 | 1 | 8.062 | 7.115 | - | 209.2 | - | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 4.637 | 5.143 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bisecting-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bisecting-kmeans.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7263.0 | 7263.0..7263.0 | 1 | - | - | - | 8151.4 | - | n_clusters=8, silhouette=0.118345 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8061.7 | 8061.7..8061.7 | 1 | - | - | - | 8151.7 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.118345 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1792.3 | 1792.3..1792.3 | 1 | 4.052 | 4.498 | - | 2830.4 | - | ari_vs_ours=0.804388, n_clusters=8, silhouette=0.090241 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'lloyd', 'bisecting_strategy': 'biggest_inertia', 'init': 'random', 'max_iter': 300, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "lloyd" | "lloyd" | "lloyd" |
| init | "random" | "random" | "random" |
| max_iter | 300 | 300 | 300 |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| seed | 7 | 7 | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 44.4 | 44.4..44.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 43.5 | 43.5..43.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 58.3 | 58.3..58.3 | 1 | 0.761 | 0.746 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### bisecting-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bisecting-kmeans.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 747.9 | 747.9..747.9 | 1 | - | - | - | 667.7 | - | n_clusters=8, silhouette=0.155357 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 728.8 | 728.8..728.8 | 1 | - | - | - | 668.0 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.155357 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 336.6 | 336.6..336.6 | 1 | 2.222 | 2.165 | - | 252.2 | - | ari_vs_ours=0.403522, n_clusters=8, silhouette=0.133742 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'lloyd', 'bisecting_strategy': 'biggest_inertia', 'init': 'random', 'max_iter': 300, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "lloyd" | "lloyd" | "lloyd" |
| init | "random" | "random" | "random" |
| max_iter | 300 | 300 | 300 |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| seed | 7 | 7 | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.2 | 4.2..4.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.8 | 12.8..12.8 | 1 | 0.304 | 0.326 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.enet-cv.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 277391.0 | 277391.0..277391.0 | 1 | - | - | - | 1946.9 | - | finite=True, r2=0.316583, rmse=0.690563 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 326333.7 | 326333.7..326333.7 | 1 | - | - | - | 1949.1 | - | finite=True, r2=0.316583, rmse=0.690563 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9181.0 | 9181.0..9181.0 | 1 | 30.214 | 35.545 | - | 6407.2 | - | finite=True, r2=0.317292, rmse=0.690205 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| l1_ratio | 0.5 | 0.5 | 0.5 |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 14.1 | 14.1..14.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 14.1 | 14.1..14.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.9 | 5.9..5.9 | 1 | 2.377 | 2.380 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.enet-cv.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6515.8 | 6515.8..6515.8 | 1 | - | - | - | 214.2 | - | finite=True, r2=0.909002, rmse=4.804540 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5854.4 | 5854.4..5854.4 | 1 | - | - | - | 200.8 | - | finite=True, r2=0.909002, rmse=4.804540 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 215.2 | 215.2..215.2 | 1 | 30.284 | 27.210 | - | 232.4 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| l1_ratio | 0.5 | 0.5 | 0.5 |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 5.434 | 7.501 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### huber / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.huber.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 113575.1 | 113575.1..113575.1 | 1 | - | - | - | 2708.4 | - | finite=True, r2=-0.009166, rmse=0.839154 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 94320.5 | 94320.5..94320.5 | 1 | - | - | - | 2281.6 | - | finite=True, r2=-0.008471, rmse=0.838865 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 67600.4 | 67600.4..67600.4 | 1 | 1.680 | 1.395 | - | 3603.7 | - | finite=True, r2=-0.010176, rmse=0.839574 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'epsilon': 1.35, 'fit_intercept': True, 'max_iter': 100, 'tol': 1e-05}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 1.35 | 1.35 | 1.35 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-05 | 1e-05 | 1e-05 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.5 | 13.5..13.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 14.2 | 14.2..14.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 22.3 | 22.3..22.3 | 1 | 0.603 | 0.636 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### huber / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.huber.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 48366.7 | 48366.7..48366.7 | 1 | - | - | - | 246.1 | - | finite=True, r2=0.900215, rmse=5.031176 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 345657.8 | 345657.8..345657.8 | 1 | - | - | - | 244.6 | - | finite=True, r2=0.900214, rmse=5.031205 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2710.0 | 2710.0..2710.0 | 1 | 17.847 | 127.547 | - | 266.5 | - | finite=True, r2=0.900215, rmse=5.031163 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'epsilon': 1.35, 'fit_intercept': True, 'max_iter': 100, 'tol': 1e-05}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 1.35 | 1.35 | 1.35 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 1e-05 | 1e-05 | 1e-05 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.4 | 1.4..1.4 | 1 | 1.565 | 1.590 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### isotonic / istella (rows full, shape X 1000000; Xq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.isotonic.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1278.3 | 1278.3..1278.3 | 1 | - | - | - | 1207.9 | - | finite=True, r2=0.187985, rmse=0.752735 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1267.3 | 1267.3..1267.3 | 1 | - | - | - | 1253.4 | - | finite=True, r2=0.187985, rmse=0.752735 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 47.2 | 47.2..47.2 | 1 | 27.089 | 26.856 | - | 1092.6 | - | finite=True, r2=0.187985, rmse=0.752735 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'increasing': True, 'out_of_bounds': 'clip'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.1 | 8.1..8.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.2 | 8.2..8.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.8 | 2.8..2.8 | 1 | 2.913 | 2.957 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### isotonic / taxi (rows full, shape X 1000000; Xq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.isotonic.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1858.1 | 1858.1..1858.1 | 1 | - | - | - | 345.5 | - | finite=True, r2=0.897069, rmse=5.109874 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1810.8 | 1810.8..1810.8 | 1 | - | - | - | 363.6 | - | finite=True, r2=0.897069, rmse=5.109874 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 109.9 | 109.9..109.9 | 1 | 16.904 | 16.473 | - | 240.6 | - | finite=True, r2=0.897069, rmse=5.109874 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'increasing': True, 'out_of_bounds': 'clip'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 35.5 | 35.5..35.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 38.2 | 38.2..38.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.3 | 5.3..5.3 | 1 | 6.724 | 7.243 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17615.2 | 17615.2..17615.2 | 1 | - | - | - | 1933.0 | - | finite=True, r2=-1.360350, rmse=1.283360 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12903.5 | 12903.5..12903.5 | 1 | - | - | - | 1934.7 | - | finite=True, r2=-1.632561, rmse=1.355344 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "OverflowError('int too large to convert to float')", "event": "error", "stage": "round 0"}) (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host not sampled; GPU not sampled

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.7 | 13.7..13.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 14.3 | 14.3..14.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "OverflowError('int too large to convert to float')", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 250.2 | 250.2..250.2 | 1 | - | - | - | 175.5 | - | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 203.1 | 203.1..203.1 | 1 | - | - | - | 179.9 | - | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 46.9 | 46.9..46.9 | 1 | 5.340 | 4.334 | - | 197.6 | - | finite=True, r2=0.908988, rmse=4.804917 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.5 | 0.5..0.5 | 1 | 4.572 | 4.404 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-cv.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 270615.9 | 270615.9..270615.9 | 1 | - | - | - | 1947.9 | - | finite=True, r2=0.310329, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 319794.0 | 319794.0..319794.0 | 1 | - | - | - | 1946.3 | - | finite=True, r2=0.310329, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7100.6 | 7100.6..7100.6 | 1 | 38.111 | 45.037 | - | 6555.8 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 26.1 | 26.1..26.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.6 | 15.6..15.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.9 | 5.9..5.9 | 1 | 4.443 | 2.652 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-cv.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6492.2 | 6492.2..6492.2 | 1 | - | - | - | 219.6 | - | finite=True, r2=0.909059, rmse=4.803051 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5839.5 | 5839.5..5839.5 | 1 | - | - | - | 215.0 | - | finite=True, r2=0.909059, rmse=4.803051 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 225.0 | 225.0..225.0 | 1 | 28.857 | 25.956 | - | 227.6 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.0 | 3.0..3.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.8 | 2.8..2.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 7.140 | 6.796 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-lars.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13463.0 | 13463.0..13463.0 | 1 | - | - | - | 1932.8 | - | finite=True, r2=0.310330, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8993.4 | 8993.4..8993.4 | 1 | - | - | - | 1937.3 | - | finite=True, r2=0.310330, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 329.5 | 329.5..329.5 | 1 | 40.861 | 27.295 | - | 1919.5 | - | finite=True, r2=0.311104, rmse=0.693326 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 | 0.01 |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| max_iter | 500 | 500 | 500 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 14.2 | 14.2..14.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 13.9 | 13.9..13.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.0 | 6.0..6.0 | 1 | 2.387 | 2.335 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-lars.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 254.5 | 254.5..254.5 | 1 | - | - | - | 178.5 | - | finite=True, r2=0.908996, rmse=4.804699 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 203.0 | 203.0..203.0 | 1 | - | - | - | 180.3 | - | finite=True, r2=0.908996, rmse=4.804698 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 45.4 | 45.4..45.4 | 1 | 5.609 | 4.474 | - | 197.2 | - | finite=True, r2=0.909003, rmse=4.804527 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 | 0.01 |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| max_iter | 500 | 500 | 500 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.5 | 0.5..0.5 | 1 | 4.352 | 4.498 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### meanshift / istella (rows full, shape X 10000x220; Xq 100000x220; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.meanshift.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8510.7 | 8510.7..8510.7 | 1 | - | - | - | 1187.2 | - | n_clusters=12, silhouette=0.403452 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8055.3 | 8055.3..8055.3 | 1 | - | - | - | 1193.0 | - | ari_vs_ours=1.000000, n_clusters=12, silhouette=0.403452 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 595.5 | 595.5..595.5 | 1 | 14.293 | 13.528 | - | 1110.6 | - | ari_vs_ours=1.000000, n_clusters=12, silhouette=0.403452 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bin_seeding': True, 'cluster_all': True, 'max_iter': 300, 'min_bin_freq': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bandwidth | 12.620826588142847 | 12.620826588142847 | 12.620826588142847 |
| max_iter | 300 | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 43.6 | 43.6..43.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 42.0 | 42.0..42.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 10.6 | 10.6..10.6 | 1 | 4.126 | 3.971 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### meanshift / taxi (rows full, shape X 10000x11; Xq 100000x11; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.meanshift.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 853.9 | 853.9..853.9 | 1 | - | - | - | 128.1 | - | n_clusters=122, silhouette=0.246631 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 781.2 | 781.2..781.2 | 1 | - | - | - | 131.0 | - | ari_vs_ours=0.999996, n_clusters=122, silhouette=0.246592 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9294.6 | 9294.6..9294.6 | 1 | 0.092 | 0.084 | - | 200.2 | - | ari_vs_ours=1.000000, n_clusters=122, silhouette=0.246631 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'bin_seeding': True, 'cluster_all': True, 'max_iter': 300, 'min_bin_freq': 1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bandwidth | 2.2313335924534736 | 2.2313335924534736 | 2.2313335924534736 |
| max_iter | 300 | 300 | 300 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.6 | 8.6..8.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 9.0 | 9.0..9.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 25.9 | 25.9..25.9 | 1 | 0.333 | 0.345 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### minibatch-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minibatch-kmeans.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 983.0 | 983.0..983.0 | 1 | - | - | - | 2840.8 | - | n_clusters=8, silhouette=0.116696 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 998.3 | 998.3..998.3 | 1 | - | - | - | 2843.4 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.116696 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 139.7 | 139.7..139.7 | 1 | 7.036 | 7.146 | - | 1113.6 | - | ari_vs_ours=0.622446, n_clusters=8, silhouette=0.111849 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 4096 | 4096 | 4096 |
| init | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 100 | 100 | 100 |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| seed | 7 | 7 | 7 |
| tol | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 54.8 | 54.8..54.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 54.5 | 54.5..54.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.6 | 6.6..6.6 | 1 | 8.265 | 8.233 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### minibatch-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minibatch-kmeans.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 135.2 | 135.2..135.2 | 1 | - | - | - | 248.2 | - | n_clusters=8, silhouette=0.138060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 134.0 | 134.0..134.0 | 1 | - | - | - | 252.1 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.138060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 30.5 | 30.5..30.5 | 1 | 4.437 | 4.398 | - | 224.0 | - | ari_vs_ours=0.525151, n_clusters=8, silhouette=0.165473 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 4096 | 4096 | 4096 |
| init | "k-means++" | "k-means++" | "k-means++" |
| max_iter | 100 | 100 | 100 |
| n_clusters | 8 | 8 | 8 |
| n_init | 1 | 1 | 1 |
| seed | 7 | 7 | 7 |
| tol | 0.0 | 0.0 | 0.0 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.9 | 2.9..2.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.9 | 0.9..0.9 | 1 | 3.459 | 3.060 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### pa-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 345788.1 | 345788.1..345788.1 | 1 | - | - | - | 1969.4 | - | accuracy=0.903700 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 255212.2 | 255212.2..255212.2 | 1 | - | - | - | 1965.5 | - | accuracy=0.903700 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10269.9 | 10269.9..10269.9 | 1 | 33.670 | 24.851 | - | 1085.9 | - | accuracy=0.890480 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'fit_intercept': True, 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 22.8 | 22.8..22.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 23.2 | 23.2..23.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.6 | 6.6..6.6 | 1 | 3.438 | 3.509 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 45901.5 | 45901.5..45901.5 | 1 | - | - | - | 228.3 | - | accuracy=0.583830 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 52191.1 | 52191.1..52191.1 | 1 | - | - | - | 238.6 | - | accuracy=0.583830 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1770.8 | 1770.8..1770.8 | 1 | 25.921 | 29.472 | - | 204.0 | - | accuracy=0.744740 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'fit_intercept': True, 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 11.4 | 11.4..11.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 11.6 | 11.6..11.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.0 | 1.0..1.0 | 1 | 10.925 | 11.058 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-reg.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 355604.6 | 355604.6..355604.6 | 1 | - | - | - | 1952.9 | - | finite=True, r2=-0.282312, rmse=0.945926 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 255976.6 | 255976.6..255976.6 | 1 | - | - | - | 1954.7 | - | finite=True, r2=-0.282312, rmse=0.945926 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14125.8 | 14125.8..14125.8 | 1 | 25.174 | 18.121 | - | 1083.7 | - | finite=True, r2=-0.128155, rmse=0.887248 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| epsilon | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 14.8 | 14.8..14.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 14.7 | 14.7..14.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.1 | 6.1..6.1 | 1 | 2.429 | 2.419 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-reg.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 47647.4 | 47647.4..47647.4 | 1 | - | - | - | 196.4 | - | finite=True, r2=0.853920, rmse=6.087406 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 53929.6 | 53929.6..53929.6 | 1 | - | - | - | 198.0 | - | finite=True, r2=0.853920, rmse=6.087406 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1824.7 | 1824.7..1824.7 | 1 | 26.113 | 29.556 | - | 206.1 | - | finite=True, r2=0.795107, rmse=7.209421 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| C | 1.0 | 1.0 | 1.0 |
| epsilon | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| loss | "epsilon_insensitive" | "epsilon_insensitive" | "epsilon_insensitive" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.7 | 0.7..0.7 | 1 | 3.201 | 2.920 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### perceptron / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.perceptron.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 260457.8 | 260457.8..260457.8 | 1 | - | - | - | 1966.9 | - | accuracy=0.882480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 209581.5 | 209581.5..209581.5 | 1 | - | - | - | 1968.8 | - | accuracy=0.882480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6168.5 | 6168.5..6168.5 | 1 | 42.224 | 33.976 | - | 1084.3 | - | accuracy=0.896130 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'early_stopping': False, 'eta0': 1.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| eta0 | 1.0 | 1.0 | 1.0 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| max_iter | 20 | 20 | 20 |
| penalty | null | null | null |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast penalty: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu penalty: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 22.8 | 22.8..22.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 25.8 | 25.8..25.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.5 | 6.5..6.5 | 1 | 3.482 | 3.951 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### perceptron / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.perceptron.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 45701.0 | 45701.0..45701.0 | 1 | - | - | - | 228.7 | - | accuracy=0.465380 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 52684.0 | 52684.0..52684.0 | 1 | - | - | - | 234.6 | - | accuracy=0.465380 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1409.5 | 1409.5..1409.5 | 1 | 32.423 | 37.378 | - | 204.2 | - | accuracy=0.750520 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'early_stopping': False, 'eta0': 1.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| eta0 | 1.0 | 1.0 | 1.0 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| max_iter | 20 | 20 | 20 |
| penalty | null | null | null |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast penalty: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu penalty: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 11.5 | 11.5..11.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 11.6 | 11.6..11.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.1 | 1.1..1.1 | 1 | 10.286 | 10.378 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 457347.9 | 457347.9..457347.9 | 1 | - | - | - | 1251.0 | - | finite=False, r2=nan, rmse=nan | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 424418.6 | 424418.6..424418.6 | 1 | - | - | - | 1253.5 | - | finite=False, r2=nan, rmse=nan | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.105e+06 | 1.105e+06..1.105e+06 | 1 | 0.414 | 0.384 | - | 3460.5 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 5000 | 5000 | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "highs" | "highs" | "highs" |
| tol | 0.0001 | 0.0001 | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.3 | 13.3..13.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 13.2 | 13.2..13.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 22.6 | 22.6..22.6 | 1 | 0.587 | 0.585 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 64540.1 | 64540.1..64540.1 | 1 | - | - | - | 133.0 | - | finite=True, r2=0.900039, rmse=5.035615 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 43810.4 | 43810.4..43810.4 | 1 | - | - | - | 135.9 | - | finite=True, r2=0.899875, rmse=5.039731 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 267425.9 | 267425.9..267425.9 | 1 | 0.241 | 0.164 | - | 369.5 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows: None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 5000 | 5000 | - |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "highs" | "highs" | "highs" |
| tol | 0.0001 | 0.0001 | - |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.6 | 1.6..1.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.6 | 1.6..1.6 | 1 | 1.367 | 0.989 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-clf.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12951.1 | 12951.1..12951.1 | 1 | - | - | - | 1951.1 | - | accuracy=0.894330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8927.7 | 8927.7..8927.7 | 1 | - | - | - | 1953.2 | - | accuracy=0.894330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9421.9 | 9421.9..9421.9 | 1 | 1.375 | 0.948 | - | 9486.0 | - | accuracy=0.910540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_intercept': True, 'positive': False, 'random_state': 7, 'solver': 'auto', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| max_iter | null | null | null |
| positive | false | false | false |
| seed | 7 | 7 | 7 |
| solver | "auto" | "auto" | "auto" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_iter: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_iter: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 22.1 | 22.1..22.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 22.7 | 22.7..22.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.4 | 6.4..6.4 | 1 | 3.436 | 3.530 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-clf.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 740.8 | 740.8..740.8 | 1 | - | - | - | 224.0 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 639.9 | 639.9..639.9 | 1 | - | - | - | 224.6 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 100.6 | 100.6..100.6 | 1 | 7.367 | 6.364 | - | 212.6 | - | accuracy=0.763580 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_intercept': True, 'positive': False, 'random_state': 7, 'solver': 'auto', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| max_iter | null | null | null |
| positive | false | false | false |
| seed | 7 | 7 | 7 |
| solver | "auto" | "auto" | "auto" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast max_iter: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu max_iter: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 12.7 | 12.7..12.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 11.7 | 11.7..11.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.1 | 1.1..1.1 | 1 | 11.576 | 10.633 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-cv.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 191217.8 | 191217.8..191217.8 | 1 | - | - | - | 8630.0 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| fit_intercept | true | true | true |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | Xq | - | 5.8 | 5.8..5.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-cv.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 1258.8 | 1258.8..1258.8 | 1 | - | - | - | 258.1 | - | finite=True, r2=0.908988, rmse=4.804917 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| cv | 5 | 5 | 5 |
| fit_intercept | true | true | true |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | Xq | - | 0.6 | 0.6..0.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-ocsvm / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.istella.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 281121.6 | 281121.6..281121.6 | 1 | - | - | - | 1959.7 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 212123.1 | 212123.1..212123.1 | 1 | - | - | - | 1956.6 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6196.8 | 6196.8..6196.8 | 1 | 45.366 | 34.231 | - | 1084.0 | - | fraction_flagged=0.093340, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'average': False, 'eta0': 0.0, 'fit_intercept': True, 'learning_rate': 'optimal', 'max_iter': 20, 'nu': 0.1, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eta0 | 0.0 | 0.0 | 0.0 |
| fit_intercept | true | true | true |
| learning_rate | "optimal" | "optimal" | "optimal" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 23.8 | 23.8..23.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 32.9 | 32.9..32.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.2 | 6.2..6.2 | 1 | 3.838 | 5.310 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-ocsvm / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.taxi.rows-full.log`, ran on ip-172-31-45-205.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 46520.0 | 46520.0..46520.0 | 1 | - | - | - | 197.5 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 54239.0 | 54239.0..54239.0 | 1 | - | - | - | 199.3 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1313.8 | 1313.8..1313.8 | 1 | 35.410 | 41.285 | - | 197.1 | - | fraction_flagged=0.007020, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'average': False, 'eta0': 0.0, 'fit_intercept': True, 'learning_rate': 'optimal', 'max_iter': 20, 'nu': 0.1, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eta0 | 0.0 | 0.0 | 0.0 |
| fit_intercept | true | true | true |
| learning_rate | "optimal" | "optimal" | "optimal" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.9 | 7.9..7.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.3 | 8.3..8.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.8 | 0.8..0.8 | 1 | 10.318 | 10.755 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML and cuVS: CUDA only; no Apple build exists.
- Classical, wave 2, not planned on this vendor: faiss-gpu: CUDA only; faiss-cpu is the arm on this box.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU tier: `--no-cpu-arm` was passed, so no `ours-cpu` arm ran.
- Our CPU tier, no ours-cpu arm: neural lm-host-train-step, lm-infer, mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: its `ours` arm already IS the CPU path (the public *Inference class runs on the host binding).
- Our CPU tier: a GBDT configuration the host side does not restate refuses by name in its ours-cpu cell (python/mojolearn/host_surface.py NO_CPU_PATH lists them), and a FAST-only run (`--modes fast`) has no ours-cpu arm: the host bindings build IDENTICAL only.
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba opponents are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels, which the board does not install; a Mamba ratio here is against a reference implementation, not a deployment kernel.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on MPS accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only
- Neural, not planned on this vendor: gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)
- Neural, not planned on this vendor: gemm-int8: torch._int_mm is a CUDA kernel; torch on MPS has no int8 matmul, so ours races alone

