# mojolearn benchmark board

Generated 2026-09-30T16:54:42Z from `board.json` (schema `mojolearn-bench-board/1`).

## Box

| field | value |
|---|---|
| vendor / API | apple / metal |
| GPU | Apple M3 Ultra |
| GPU driver | macOS 26.6 |
| CPU | Apple M3 Ultra (28 logical cores) |
| memory bytes | 274877906944 |
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

Races: 1 planned, 161 done, 5 failed, 0 pending. Cells: 557 (HOST-MEMORY 9, REFUSED 26, ok 522).

Inference cells: 377 (REFUSED 18, ok 359).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | additive-chi2 | istella | kernel_rel_error (lower is better) | 0.087730 | 0.087730 | - | sklearn-cpu 0.087730 |
| algos | additive-chi2 | taxi | kernel_rel_error (lower is better) | 0.093892 | 0.093892 | - | sklearn-cpu 0.093892 |
| algos | affinity-prop | istella | n_clusters | 342 | 342 | - | sklearn-cpu 342 |
| algos | affinity-prop | istella | silhouette (higher is better) | 0.089763 | 0.089763 | - | sklearn-cpu 0.089763 |
| algos | affinity-prop | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | affinity-prop | taxi | n_clusters | 272 | 272 | - | sklearn-cpu 272 |
| algos | affinity-prop | taxi | silhouette (higher is better) | 0.184644 | 0.184644 | - | sklearn-cpu 0.184644 |
| algos | affinity-prop | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | ard | istella | r2 (higher is better) | -0.123501 | -0.122912 | - | sklearn-cpu 0.327436 |
| algos | ard | istella | rmse (lower is better) | 0.885416 | 0.885183 | - | sklearn-cpu 0.685058 |
| algos | ard | taxi | r2 (higher is better) | 0.909193 | 0.909193 | - | sklearn-cpu 0.909190 |
| algos | ard | taxi | rmse (lower is better) | 4.799513 | 4.799513 | - | sklearn-cpu 4.799575 |
| algos | bayesian-gmm | istella | mean_log_likelihood (higher is better) | 175.561762 | 175.322029 | - | sklearn-cpu 201.770716 |
| algos | bayesian-gmm | taxi | mean_log_likelihood (higher is better) | 4.895586 | 4.895581 | - | sklearn-cpu 6.178321 |
| algos | bayesian-ridge | istella | r2 (higher is better) | -41688.617698 | -41643.670747 | - | sklearn-cpu -890.186917 |
| algos | bayesian-ridge | istella | rmse (lower is better) | 170.558899 | 170.466932 | - | sklearn-cpu 24.937036 |
| algos | bayesian-ridge | taxi | r2 (higher is better) | 0.908981 | 0.908981 | - | sklearn-cpu 0.908983 |
| algos | bayesian-ridge | taxi | rmse (lower is better) | 4.805108 | 4.805109 | - | sklearn-cpu 4.805052 |
| algos | bisecting-kmeans | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | bisecting-kmeans | istella | silhouette (higher is better) | 0.118345 | 0.118345 | - | sklearn-cpu 0.096414 |
| algos | bisecting-kmeans | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.692929 |
| algos | bisecting-kmeans | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | bisecting-kmeans | taxi | silhouette (higher is better) | 0.155357 | 0.155357 | - | sklearn-cpu 0.128033 |
| algos | bisecting-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.502256 |
| algos | enet-cv | istella | r2 (higher is better) | 0.316583 | 0.316583 | - | sklearn-cpu 0.317292 |
| algos | enet-cv | istella | rmse (lower is better) | 0.690563 | 0.690563 | - | sklearn-cpu 0.690205 |
| algos | enet-cv | taxi | r2 (higher is better) | 0.909002 | 0.909002 | - | sklearn-cpu 0.909004 |
| algos | enet-cv | taxi | rmse (lower is better) | 4.804540 | 4.804540 | - | sklearn-cpu 4.804486 |
| algos | huber | istella | r2 (higher is better) | -0.008227 | -0.009166 | - | sklearn-cpu -0.010176 |
| algos | huber | istella | rmse (lower is better) | 0.838764 | 0.839154 | - | sklearn-cpu 0.839574 |
| algos | huber | taxi | r2 (higher is better) | 0.900217 | 0.900215 | - | sklearn-cpu 0.900215 |
| algos | huber | taxi | rmse (lower is better) | 5.031119 | 5.031176 | - | sklearn-cpu 5.031163 |
| algos | isotonic | istella | r2 (higher is better) | 0.187985 | 0.187985 | - | sklearn-cpu 0.187985 |
| algos | isotonic | istella | rmse (lower is better) | 0.752735 | 0.752735 | - | sklearn-cpu 0.752735 |
| algos | isotonic | taxi | r2 (higher is better) | 0.897069 | 0.897069 | - | sklearn-cpu 0.897069 |
| algos | isotonic | taxi | rmse (lower is better) | 5.109874 | 5.109874 | - | sklearn-cpu 5.109874 |
| algos | knn-imputer | taxi | masked_rmse | 6.151696 | 6.151696 | - | sklearn-cpu 5.256719 |
| algos | knn-imputer | taxi | max_abs_diff_vs_sklearn | 29.000000 | 29.000000 | - | sklearn-cpu - |
| algos | label-propagation | istella | accuracy (higher is better) | - | - | - | sklearn-cpu 0.905500 |
| algos | label-propagation | taxi | accuracy (higher is better) | - | - | - | sklearn-cpu 0.701600 |
| algos | label-spreading | istella | accuracy (higher is better) | - | - | - | sklearn-cpu 0.904450 |
| algos | label-spreading | taxi | accuracy (higher is better) | - | - | - | sklearn-cpu 0.676400 |
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
| algos | lof | istella | fraction_flagged | 0.033610 | 0.033610 | - | sklearn-cpu 0.033610 |
| algos | lof | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | lof | taxi | fraction_flagged | 0.008960 | 0.008960 | - | sklearn-cpu 0.008960 |
| algos | lof | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | logreg-cv | istella | accuracy (higher is better) | 0.924520 | - | - | sklearn-cpu 0.924630 |
| algos | logreg-cv | istella | logloss (lower is better) | 0.181412 | - | - | sklearn-cpu 0.181351 |
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
| algos | nearest-centroid | istella | accuracy (higher is better) | 0.852610 | 0.852610 | - | sklearn-cpu 0.852610 |
| algos | nearest-centroid | istella | logloss (lower is better) | 4.299222 | 4.299221 | - | sklearn-cpu 4.117692 |
| algos | nearest-centroid | taxi | accuracy (higher is better) | 0.666750 | 0.666750 | - | sklearn-cpu 0.666750 |
| algos | nearest-centroid | taxi | logloss (lower is better) | 0.782162 | 0.782162 | - | sklearn-cpu 0.781690 |
| algos | ocsvm | istella | fraction_flagged | 0.078300 | 0.078300 | - | sklearn-cpu 0.078300 |
| algos | ocsvm | istella | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | ocsvm | taxi | fraction_flagged | 0.136100 | 0.136100 | - | sklearn-cpu 0.136100 |
| algos | ocsvm | taxi | jaccard_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | optics | istella | n_clusters | 20 | 20 | - | sklearn-cpu 20 |
| algos | optics | istella | silhouette (higher is better) | -0.287356 | -0.287356 | - | sklearn-cpu -0.285806 |
| algos | optics | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.984603 |
| algos | optics | taxi | n_clusters | 127 | 127 | - | sklearn-cpu 127 |
| algos | optics | taxi | silhouette (higher is better) | -0.353359 | -0.353359 | - | sklearn-cpu -0.353359 |
| algos | optics | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | pa-clf | istella | accuracy (higher is better) | 0.903700 | 0.903700 | - | sklearn-cpu 0.890480 |
| algos | pa-clf | taxi | accuracy (higher is better) | 0.583830 | 0.583830 | - | sklearn-cpu 0.744740 |
| algos | pa-reg | istella | r2 (higher is better) | -0.282312 | -0.282312 | - | sklearn-cpu -0.128155 |
| algos | pa-reg | istella | rmse (lower is better) | 0.945926 | 0.945926 | - | sklearn-cpu 0.887248 |
| algos | pa-reg | taxi | r2 (higher is better) | 0.853920 | 0.853920 | - | sklearn-cpu 0.795107 |
| algos | pa-reg | taxi | rmse (lower is better) | 6.087406 | 6.087406 | - | sklearn-cpu 7.209421 |
| algos | perceptron | istella | accuracy (higher is better) | 0.882480 | 0.882480 | - | sklearn-cpu 0.896130 |
| algos | perceptron | taxi | accuracy (higher is better) | 0.465380 | 0.465380 | - | sklearn-cpu 0.750520 |
| algos | poisson | taxi | r2 (higher is better) | 0.035965 | 0.035965 | - | sklearn-cpu 0.036205 |
| algos | poisson | taxi | rmse (lower is better) | 15.638068 | 15.638069 | - | sklearn-cpu 15.636127 |
| algos | poly-count-sketch | istella | kernel_rel_error (lower is better) | 0.040849 | 0.040849 | - | sklearn-cpu 0.040849 |
| algos | poly-count-sketch | taxi | kernel_rel_error (lower is better) | 0.096596 | 0.096596 | - | sklearn-cpu 0.096596 |
| algos | quantile | istella | r2 (higher is better) | nan | nan | - | sklearn-cpu -0.044780 |
| algos | quantile | istella | rmse (lower is better) | nan | nan | - | sklearn-cpu 0.853833 |
| algos | quantile | taxi | r2 (higher is better) | 0.899875 | 0.900039 | - | sklearn-cpu 0.899678 |
| algos | quantile | taxi | rmse (lower is better) | 5.039731 | 5.035615 | - | sklearn-cpu 5.044706 |
| algos | radius-neighbors | istella | count_agreement_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu - |
| algos | radius-neighbors | istella | neighbors_total | 1220718 | 1220718 | - | sklearn-cpu 1220718 |
| algos | radius-neighbors | taxi | count_agreement_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu - |
| algos | radius-neighbors | taxi | neighbors_total | 31 | 31 | - | sklearn-cpu 31 |
| algos | ridge-clf | istella | accuracy (higher is better) | 0.894330 | 0.894330 | - | sklearn-cpu 0.910540 |
| algos | ridge-clf | taxi | accuracy (higher is better) | 0.763570 | 0.763570 | - | sklearn-cpu 0.763580 |
| algos | ridge-cv | istella | r2 (higher is better) | - | - | - | sklearn-cpu 0.328683 |
| algos | ridge-cv | istella | rmse (lower is better) | - | - | - | sklearn-cpu 0.684423 |
| algos | ridge-cv | taxi | r2 (higher is better) | - | - | - | sklearn-cpu 0.908988 |
| algos | ridge-cv | taxi | rmse (lower is better) | - | - | - | sklearn-cpu 4.804917 |
| algos | sgd-clf | istella | accuracy (higher is better) | 0.901150 | 0.901150 | - | sklearn-cpu 0.910200 |
| algos | sgd-clf | taxi | accuracy (higher is better) | 0.766410 | 0.766410 | - | sklearn-cpu 0.752520 |
| algos | sgd-ocsvm | istella | fraction_flagged | 0.000000 | 0.000000 | - | sklearn-cpu 0.093340 |
| algos | sgd-ocsvm | istella | jaccard_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu 1.000000 |
| algos | sgd-ocsvm | taxi | fraction_flagged | 0.000000 | 0.000000 | - | sklearn-cpu 0.007020 |
| algos | sgd-ocsvm | taxi | jaccard_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu 1.000000 |
| algos | sgd-reg | istella | r2 (higher is better) | -3.459e+24 | -3.459e+24 | - | sklearn-cpu -2.197e+24 |
| algos | sgd-reg | istella | rmse (lower is better) | 1.554e+12 | 1.554e+12 | - | sklearn-cpu 1.238e+12 |
| algos | sgd-reg | taxi | r2 (higher is better) | 0.868127 | 0.868127 | - | sklearn-cpu 0.880681 |
| algos | sgd-reg | taxi | rmse (lower is better) | 5.783813 | 5.783813 | - | sklearn-cpu 5.501638 |
| algos | skewed-chi2 | istella | kernel_rel_error (lower is better) | 0.671898 | 0.671898 | - | sklearn-cpu 0.671899 |
| algos | skewed-chi2 | taxi | kernel_rel_error (lower is better) | 0.037749 | 0.037749 | - | sklearn-cpu 0.037749 |
| classical | dbscan | istella | n_clusters | 40131 | 40131 | - | sklearn-cpu 40131 |
| classical | dbscan | istella | noise_fraction | 0.219391 | 0.219391 | - | sklearn-cpu 0.219391 |
| classical | dbscan | istella | rows | 1000000 | 1000000 | - | sklearn-cpu 1000000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | taxi | n_clusters | 36 | - | - | sklearn-cpu - |
| classical | dbscan | taxi | noise_fraction | 0.000174 | - | - | sklearn-cpu - |
| classical | dbscan | taxi | rows | 1000000 | - | - | sklearn-cpu - |
| classical | hdbscan | istella | n_clusters | - | - | - | sklearn-cpu 52 |
| classical | hdbscan | istella | noise_fraction | - | - | - | sklearn-cpu 0.252570 |
| classical | hdbscan | istella | rows | - | - | - | sklearn-cpu 100000 |
| classical | hdbscan | taxi | n_clusters | 160 | - | - | sklearn-cpu 161 |
| classical | hdbscan | taxi | noise_fraction | 0.142220 | - | - | sklearn-cpu 0.134620 |
| classical | hdbscan | taxi | rows | 100000 | - | - | sklearn-cpu 100000 |
| classical | kde | istella | mean_log_likelihood (higher is better) | -222.270586 | -222.270586 | - | sklearn-cpu -226.977403 |
| classical | kde | istella | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | -14.826460 | -14.826460 | - | sklearn-cpu -14.826437 |
| classical | kde | taxi | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kmeans | istella | inertia (lower is better) | 6.051e+17 | 6.051e+17 | - | sklearn-cpu 5.959e+17; torch-gpu 5.991e+17 |
| classical | kmeans | istella | inertia_over_ours | 1.000000 | 1.000000 | - | sklearn-cpu 0.984774; torch-gpu 0.990156 |
| classical | kmeans | istella | n_iter | 33 | 33 | - | sklearn-cpu 36; torch-gpu 68 |
| classical | kmeans | taxi | inertia (lower is better) | 3.093e+08 | 3.093e+08 | - | sklearn-cpu 3.166e+08; torch-gpu 3.06e+08 |
| classical | kmeans | taxi | inertia_over_ours | 1.000000 | 1.000000 | - | sklearn-cpu 1.023705; torch-gpu 0.989329 |
| classical | kmeans | taxi | n_iter | 91 | 91 | - | sklearn-cpu 100; torch-gpu 71 |
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
| classical | svc | taxi | n_support | 5525 | 5527 | - | sklearn-cpu 5672 |
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
| classical2 | gmm | istella | mean_log_likelihood (higher is better) | 200.794469 | 200.794403 | - | sklearn-cpu 200.776340 |
| classical2 | gmm | istella | n_iter | 24 | 24 | - | sklearn-cpu 30 |
| classical2 | gmm | taxi | bic (lower is better) | - | -3.67e+06 | - | sklearn-cpu - |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | - | 12.861940 | - | sklearn-cpu - |
| classical2 | gmm | taxi | n_iter | - | 32 | - | sklearn-cpu - |
| classical2 | gpc | istella | accuracy (higher is better) | 0.901333 | 0.901333 | - | sklearn-cpu 0.901333 |
| classical2 | gpc | istella | logloss (lower is better) | 0.232592 | 0.232590 | - | sklearn-cpu 0.232597 |
| classical2 | gpc | istella | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | gpc | taxi | accuracy (higher is better) | 0.761000 | 0.761000 | - | sklearn-cpu 0.761000 |
| classical2 | gpc | taxi | logloss (lower is better) | 0.541355 | 0.541286 | - | sklearn-cpu 0.541358 |
| classical2 | gpc | taxi | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | gpr | istella | mean_log_predictive_density (higher is better) | -9.285319 | -9.285754 | - | sklearn-cpu -9.287148 |
| classical2 | gpr | istella | r2 (higher is better) | 0.235374 | 0.235346 | - | sklearn-cpu 0.235368 |
| classical2 | gpr | istella | rmse (lower is better) | 0.760426 | 0.760439 | - | sklearn-cpu 0.760428 |
| classical2 | gpr | taxi | mean_log_predictive_density (higher is better) | -311.454431 | -311.458394 | - | sklearn-cpu -311.539594 |
| classical2 | gpr | taxi | r2 (higher is better) | 0.889630 | 0.889630 | - | sklearn-cpu 0.889629 |
| classical2 | gpr | taxi | rmse (lower is better) | 5.041637 | 5.041639 | - | sklearn-cpu 5.041653 |
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
| classical2 | logreg | istella | accuracy (higher is better) | 0.924560 | 0.924540 | - | sklearn-cpu 0.924470 |
| classical2 | logreg | istella | logloss (lower is better) | 0.181242 | 0.181245 | - | sklearn-cpu 0.181337 |
| classical2 | logreg | istella | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | 0.763310 | 0.763340 | - | sklearn-cpu 0.763320 |
| classical2 | logreg | taxi | logloss (lower is better) | 0.538987 | 0.538984 | - | sklearn-cpu 0.538980 |
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
| classical2 | svr | istella | r2 (higher is better) | 0.318258 | 0.318258 | - | sklearn-cpu 0.318248 |
| classical2 | svr | istella | rmse (lower is better) | 0.680816 | 0.680816 | - | sklearn-cpu 0.680821 |
| classical2 | svr | taxi | r2 (higher is better) | 0.767550 | 0.767551 | - | sklearn-cpu 0.767550 |
| classical2 | svr | taxi | rmse (lower is better) | 7.680409 | 7.680395 | - | sklearn-cpu 7.680414 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | 0.999992 | 0.999992 | - | sklearn-cpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | 0.002554 | 0.002554 | - | sklearn-cpu 0.000122 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | 0.999965 | 0.999965 | - | sklearn-cpu 0.999965 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | 0.003257 | 0.003257 | - | sklearn-cpu 0.003257 |
| classical2 | umap | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.982091 | 0.979906 | - | umap-learn-cpu 0.978822; umap-learn-cpu-unseeded 0.977078 |
| classical2 | umap | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.991488 | 0.990480 | - | umap-learn-cpu 0.989525; umap-learn-cpu-unseeded 0.991876 |
| neural | gemm-bf16 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 1.155e-07 | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-bf16 | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 0.999023; torch-compile-bf16 0.999023 |
| neural | gemm-bf16 | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-bf16 0.002759; torch-compile-bf16 0.002759 |
| neural | gemm-int8 | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 0.000000 | - | - |
| neural | gemm | gaussian | max_rel_err_vs_fp64 (lower is better) | - | 2.399e-07 | - | torch-eager-fp32 2.843e-06; torch-compile-fp32 2.843e-06; torch-eager-bf16 0.003752; torch-compile-bf16 0.003752 |
| neural | gemm | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 0.001038; torch-compile-fp32 0.001038; torch-eager-bf16 1.358337; torch-compile-bf16 1.358337 |
| neural | gemm | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.866e-06; torch-compile-fp32 2.866e-06; torch-eager-bf16 0.003751; torch-compile-bf16 0.003751 |
| neural | lm-forward | bytes | mean_nll (lower is better) | - | 9.018733 | - | torch-eager-fp32 9.018733; torch-compile-fp32 9.018733; torch-eager-bf16 9.018650; torch-compile-bf16 9.018662 |
| neural | lm-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.252e-06; torch-compile-fp32 1.296e-06; torch-eager-bf16 0.006710; torch-compile-bf16 0.006710 |
| neural | lm-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.718e-06; torch-compile-fp32 1.78e-06; torch-eager-bf16 0.009211; torch-compile-bf16 0.009211 |
| neural | lm-host-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.017858 | - | torch-cpu-compile-fp32 9.017857; torch-cpu-compile-bf16 9.017725; torch-cpu-eager-bf16 9.017766; torch-cpu-eager-fp32 9.017857 |
| neural | lm-host-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.367768 | - | torch-cpu-compile-fp32 8.367764; torch-cpu-compile-bf16 8.367495; torch-cpu-eager-bf16 8.368875; torch-cpu-eager-fp32 8.367766 |
| neural | lm-host-train-step | bytes | steps | - | 2 | - | torch-cpu-compile-fp32 2; torch-cpu-compile-bf16 2; torch-cpu-eager-bf16 2; torch-cpu-eager-fp32 2 |
| neural | lm-host-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-compile-fp32 9.537e-07; torch-cpu-compile-bf16 0.0001326; torch-cpu-eager-bf16 9.155e-05; torch-cpu-eager-fp32 9.537e-07 |
| neural | lm-host-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-cpu-compile-fp32 3.815e-06; torch-cpu-compile-bf16 0.0002737; torch-cpu-eager-bf16 0.001106; torch-cpu-eager-fp32 1.907e-06 |
| neural | lm-infer | bytes | mean_nll (lower is better) | - | 9.017857 | - | torch-cpu-compile-fp32 9.017857; torch-cpu-compile-bf16 9.017744; torch-cpu-eager-bf16 9.017766; torch-cpu-eager-fp32 9.017857 |
| neural | lm-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 1.013e-06; torch-cpu-compile-bf16 0.006585; torch-cpu-eager-bf16 0.006131; torch-cpu-eager-fp32 9.239e-07 |
| neural | lm-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 1.391e-06; torch-cpu-compile-bf16 0.009040; torch-cpu-eager-bf16 0.008417; torch-cpu-eager-fp32 1.268e-06 |
| neural | lm-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 9.018733 | - | torch-eager-fp32 9.018733; torch-compile-fp32 9.018733; torch-eager-bf16 9.021973; torch-compile-bf16 9.018524 |
| neural | lm-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 8.422411 | - | torch-eager-fp32 8.422412; torch-compile-fp32 8.422413; torch-eager-bf16 8.418335; torch-compile-bf16 8.420197 |
| neural | lm-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | lm-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 0.000000; torch-compile-fp32 0.000000; torch-eager-bf16 0.003240; torch-compile-bf16 0.0002089 |
| neural | lm-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 9.537e-07; torch-compile-fp32 1.907e-06; torch-eager-bf16 0.004076; torch-compile-bf16 0.002214 |
| neural | mamba1-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-eager-bf16 5.15e-05 |
| neural | mamba1-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.189e-07; torch-eager-bf16 2.568e-05 |
| neural | mamba1-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 1.192e-07; torch-cpu-eager-bf16 5.15e-05 |
| neural | mamba1-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 5.948e-08; torch-cpu-eager-bf16 2.57e-05 |
| neural | mamba2-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.027e-06; torch-compile-fp32 2.027e-06; torch-eager-bf16 0.007544; torch-compile-bf16 0.007544 |
| neural | mamba2-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 6.89e-07; torch-compile-fp32 6.89e-07; torch-eager-bf16 0.002565; torch-compile-bf16 0.002565 |
| neural | mamba2-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 1.192e-06; torch-cpu-compile-bf16 0.005920; torch-cpu-eager-bf16 0.005920; torch-cpu-eager-fp32 1.192e-06 |
| neural | mamba2-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 4.233e-07; torch-cpu-compile-bf16 0.002102; torch-cpu-eager-bf16 0.002102; torch-cpu-eager-fp32 4.233e-07 |
| neural | mamba3-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 8.345e-07; torch-compile-fp32 -; torch-eager-bf16 0.002095; torch-compile-bf16 - |
| neural | mamba3-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 3.651e-07; torch-compile-fp32 -; torch-eager-bf16 0.0009164; torch-compile-bf16 - |
| neural | mamba3-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 4.768e-07; torch-cpu-compile-bf16 0.002040; torch-cpu-eager-bf16 0.002148; torch-cpu-eager-fp32 4.768e-07 |
| neural | mamba3-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 2.119e-07; torch-cpu-compile-bf16 0.0009065; torch-cpu-eager-bf16 0.0009547; torch-cpu-eager-fp32 2.119e-07 |
| neural | mlp-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 0.000000; torch-cpu-compile-fp32 0.000000; torch-cpu-eager-bf16 0.004285; torch-cpu-compile-bf16 0.004285 |
| neural | mlp-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-eager-fp32 0.000000; torch-cpu-compile-fp32 0.000000; torch-cpu-eager-bf16 0.003956; torch-cpu-compile-bf16 0.003956 |
| neural | mlp-train-step | gaussian | loss_first_step (same init and batches on every arm) | - | 1.160401 | - | torch-eager-fp32 1.160401; torch-compile-fp32 1.160401; torch-eager-bf16 1.160498; torch-compile-bf16 1.160498 |
| neural | mlp-train-step | gaussian | loss_last_step (same init and batches on every arm) | - | 1.123361 | - | torch-eager-fp32 1.123361; torch-compile-fp32 1.123361; torch-eager-bf16 1.123461; torch-compile-bf16 1.123461 |
| neural | mlp-train-step | gaussian | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 2; torch-eager-bf16 2; torch-compile-bf16 2 |
| neural | mlp-train-step | gaussian | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 3.576e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 9.656e-05; torch-compile-bf16 9.656e-05 |
| neural | mlp-train-step | gaussian | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 2.384e-07; torch-compile-fp32 2.384e-07; torch-eager-bf16 0.0001006; torch-compile-bf16 0.0001006 |
| neural | samba-forward | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-eager-fp32 5.635910; torch-compile-fp32 -; torch-eager-bf16 5.636059; torch-compile-bf16 - |
| neural | samba-forward | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 2.503e-06; torch-compile-fp32 -; torch-eager-bf16 0.017546; torch-compile-bf16 - |
| neural | samba-forward | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.407e-06; torch-compile-fp32 -; torch-eager-bf16 0.009865; torch-compile-bf16 - |
| neural | samba-infer | bytes | mean_nll (lower is better) | - | 5.635910 | - | torch-cpu-compile-fp32 5.635910; torch-cpu-compile-bf16 5.635868; torch-cpu-eager-bf16 5.636032; torch-cpu-eager-fp32 5.635910 |
| neural | samba-infer | bytes | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 3.666e-06; torch-cpu-compile-bf16 0.016022; torch-cpu-eager-bf16 0.018826; torch-cpu-eager-fp32 2.772e-06 |
| neural | samba-infer | bytes | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 2.061e-06; torch-cpu-compile-bf16 0.009008; torch-cpu-eager-bf16 0.010584; torch-cpu-eager-fp32 1.558e-06 |
| neural | samba-train-step | bytes | loss_first_step (same init and batches on every arm) | - | 5.635910 | - | torch-eager-fp32 5.635910; torch-compile-fp32 -; torch-eager-bf16 5.636006; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_step (same init and batches on every arm) | - | 4.833934 | - | torch-eager-fp32 4.833934; torch-compile-fp32 -; torch-eager-bf16 4.833773; torch-compile-bf16 - |
| neural | samba-train-step | bytes | steps | - | 2 | - | torch-eager-fp32 2; torch-compile-fp32 -; torch-eager-bf16 2; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_first_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 -; torch-eager-bf16 9.632e-05; torch-compile-bf16 - |
| neural | samba-train-step | bytes | loss_last_abs_diff_vs_ours (0 is our value exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 -; torch-eager-bf16 0.0001612; torch-compile-bf16 - |
| neural | transformer-forward | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 4.768e-07; torch-compile-fp32 4.768e-07; torch-eager-bf16 0.001819; torch-compile-bf16 0.001819 |
| neural | transformer-forward | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-eager-fp32 1.02e-07; torch-compile-fp32 1.02e-07; torch-eager-bf16 0.0003893; torch-compile-bf16 0.0003893 |
| neural | transformer-infer | gaussian | max_abs_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 9.775e-06; torch-cpu-compile-bf16 0.001819; torch-cpu-eager-bf16 0.001941; torch-cpu-eager-fp32 4.768e-07 |
| neural | transformer-infer | gaussian | max_rel_diff_vs_ours (0 is our output exactly) | - | - | - | torch-cpu-compile-fp32 2.207e-06; torch-cpu-compile-bf16 0.0004107; torch-cpu-eager-bf16 0.0004383; torch-cpu-eager-fp32 1.076e-07 |
| trees | et | istella | auc (higher is better) | 0.937987 | 0.937987 | - | sklearn-et-cpu 0.937904; lightgbm-cpu 0.948128 |
| trees | et | istella | logloss (lower is better) | 0.189989 | 0.189989 | - | sklearn-et-cpu 0.190078; lightgbm-cpu 0.197270 |
| trees | et | taxi | auc (higher is better) | 0.618907 | 0.618907 | - | sklearn-et-cpu 0.618972; lightgbm-cpu 0.611172 |
| trees | et | taxi | logloss (lower is better) | 0.526142 | 0.526142 | - | sklearn-et-cpu 0.525976; lightgbm-cpu 0.529303 |
| trees | gbdt-categorical | taxi | auc (higher is better) | 0.630212 | 0.630363 | - | catboost-cpu 0.628462; xgboost-cpu 0.631472; lightgbm-cpu 0.632665 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | 0.528575 | 0.528463 | - | catboost-cpu 0.529047; xgboost-cpu 0.528686; lightgbm-cpu 0.528094 |
| trees | gbdt-depthwise | istella | auc (higher is better) | 0.980182 | 0.979129 | - | catboost-cpu 0.983136; xgboost-cpu 0.983622 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | 0.181879 | 0.188483 | - | catboost-cpu 0.157685; xgboost-cpu 0.149263 |
| trees | gbdt-depthwise | taxi | auc (higher is better) | 0.625840 | 0.625417 | - | catboost-cpu 0.632459; xgboost-cpu 0.630969 |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | 0.530036 | 0.530232 | - | catboost-cpu 0.527789; xgboost-cpu 0.528677 |
| trees | gbdt-lossguide | istella | auc (higher is better) | 0.983790 | 0.983749 | - | catboost-cpu 0.983136; xgboost-cpu 0.983622; lightgbm-cpu 0.983778 |
| trees | gbdt-lossguide | istella | logloss (lower is better) | 0.148810 | 0.149364 | - | catboost-cpu 0.157685; xgboost-cpu 0.149263; lightgbm-cpu 0.149653 |
| trees | gbdt-lossguide | taxi | auc (higher is better) | 0.631046 | 0.631154 | - | catboost-cpu 0.632459; xgboost-cpu 0.630969; lightgbm-cpu 0.632243 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | 0.528287 | 0.528317 | - | catboost-cpu 0.527789; xgboost-cpu 0.528677; lightgbm-cpu 0.528067 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | 0.903310 | 0.903294 | - | catboost-cpu 0.907768; xgboost-cpu 0.910140; lightgbm-cpu 0.910058 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | 0.280935 | 0.281958 | - | catboost-cpu 0.258286; xgboost-cpu 0.246803; lightgbm-cpu 0.245916 |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | 0.596646 | 0.596646 | - | catboost-cpu 0.599150; xgboost-cpu 0.601128; lightgbm-cpu 0.601580 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | 1.022664 | 1.022664 | - | catboost-cpu 1.012734; xgboost-cpu 1.005128; lightgbm-cpu 1.004282 |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | 0.841358 | 0.844605 | - | catboost-cpu 0.846328; xgboost-cpu 0.872796 |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | 0.709249 | 0.711712 | - | catboost-cpu 0.713361; xgboost-cpu 0.738397 |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | 0.639753 | 0.641863 | - | catboost-cpu 0.643611; xgboost-cpu 0.670093 |
| trees | gbdt-rank-yetirank | istella | map (higher is better) | 0.814902 | 0.814902 | - | catboost-cpu 0.851990; xgboost-cpu 0.842929; lightgbm-cpu 0.858362 |
| trees | gbdt-rank-yetirank | istella | ndcg10 (higher is better) | 0.680982 | 0.680993 | - | catboost-cpu 0.726111; xgboost-cpu 0.726256; lightgbm-cpu 0.741515 |
| trees | gbdt-rank-yetirank | istella | ndcg5 (higher is better) | 0.615063 | 0.615076 | - | catboost-cpu 0.660263; xgboost-cpu 0.664249; lightgbm-cpu 0.680254 |
| trees | gbdt-symmetric-1000 | istella | auc (higher is better) | 0.975635 | 0.977075 | - | catboost-cpu 0.982309 |
| trees | gbdt-symmetric-1000 | istella | logloss (lower is better) | 0.211169 | 0.203751 | - | catboost-cpu 0.171620 |
| trees | gbdt-symmetric-1000 | taxi | auc (higher is better) | 0.623944 | 0.621310 | - | catboost-cpu 0.631642 |
| trees | gbdt-symmetric-1000 | taxi | logloss (lower is better) | 0.530645 | 0.531329 | - | catboost-cpu 0.528267 |
| trees | gbdt-symmetric | istella | auc (higher is better) | 0.975635 | 0.977075 | - | catboost-cpu 0.979899 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | 0.211169 | 0.203751 | - | catboost-cpu 0.188093 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | 0.623944 | 0.621310 | - | catboost-cpu 0.630269 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | 0.530645 | 0.531329 | - | catboost-cpu 0.528650 |
| trees | iforest | istella | auc (higher is better) | 0.830358 | 0.830358 | - | sklearn-iforest-cpu 0.827914 |
| trees | iforest | taxi | auc (higher is better) | 0.551846 | 0.551846 | - | sklearn-iforest-cpu 0.552849 |
| trees | rf | istella | auc (higher is better) | 0.945385 | 0.945385 | - | sklearn-rf-cpu 0.945284; lightgbm-cpu 0.945361 |
| trees | rf | istella | logloss (lower is better) | 0.182017 | 0.182017 | - | sklearn-rf-cpu 0.182344; lightgbm-cpu 0.195422 |
| trees | rf | taxi | auc (higher is better) | 0.617838 | 0.617838 | - | sklearn-rf-cpu 0.617678; lightgbm-cpu 0.617040 |
| trees | rf | taxi | logloss (lower is better) | 0.525953 | 0.525953 | - | sklearn-rf-cpu 0.525532; lightgbm-cpu 0.526421 |

## Inference at a glance

Batch prediction, each arm with its own fitted model from the same race; medians in ms. `FAST = IDENTICAL bits` compares our two tiers' predictions on the same rows; `CPU = IDENTICAL bits` compares our CPU tier's with our GPU IDENTICAL arm's.

| family | lane | dataset | batch | rows | ours FAST ms | ours IDENTICAL ms | FAST = IDENTICAL bits | ours CPU ms | CPU = IDENTICAL bits | opponents |
|---|---|---|---|---|---|---|---|---|---|---|
| algos | additive-chi2 | istella | Xq | - | 2.3 | 2.2 | - | - | - | sklearn-cpu 2.4 ms (IDENTICAL/arm 0.920) |
| algos | additive-chi2 | taxi | Xq | - | 1.2 | 1.3 | - | - | - | sklearn-cpu 0.3 ms (IDENTICAL/arm 4.929) |
| algos | affinity-prop | istella | Xq | - | 49.0 | 52.2 | - | - | - | sklearn-cpu 19.0 ms (IDENTICAL/arm 2.747) |
| algos | affinity-prop | taxi | Xq | - | 4.0 | 4.4 | - | - | - | sklearn-cpu 5.5 ms (IDENTICAL/arm 0.798) |
| algos | ard | istella | Xq | - | 15.1 | 14.6 | - | - | - | sklearn-cpu 4.0 ms (IDENTICAL/arm 3.685) |
| algos | ard | taxi | Xq | - | 1.0 | 0.9 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 2.538) |
| algos | bayesian-gmm | istella | Xq | - | 55.2 | 55.9 | - | - | - | sklearn-cpu 89.6 ms (IDENTICAL/arm 0.624) |
| algos | bayesian-gmm | taxi | Xq | - | 13.6 | 13.7 | - | - | - | sklearn-cpu 5.9 ms (IDENTICAL/arm 2.302) |
| algos | bayesian-ridge | istella | Xq | - | 15.8 | 23.5 | - | - | - | sklearn-cpu 3.9 ms (IDENTICAL/arm 6.099) |
| algos | bayesian-ridge | taxi | Xq | - | 1.3 | 1.4 | - | - | - | sklearn-cpu 0.5 ms (IDENTICAL/arm 2.858) |
| algos | bisecting-kmeans | istella | Xq | - | 41.8 | 22.3 | - | - | - | sklearn-cpu 60.3 ms (IDENTICAL/arm 0.369) |
| algos | bisecting-kmeans | taxi | Xq | - | 2.0 | 2.4 | - | - | - | sklearn-cpu 15.6 ms (IDENTICAL/arm 0.151) |
| algos | enet-cv | istella | Xq | - | 16.4 | 17.2 | - | - | - | sklearn-cpu 4.0 ms (IDENTICAL/arm 4.313) |
| algos | enet-cv | taxi | Xq | - | 2.2 | 2.2 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 6.238) |
| algos | huber | istella | Xq | - | 10.9 | 8.0 | - | - | - | sklearn-cpu 16.0 ms (IDENTICAL/arm 0.501) |
| algos | huber | taxi | Xq | - | 1.8 | 1.8 | - | - | - | sklearn-cpu 1.1 ms (IDENTICAL/arm 1.527) |
| algos | isotonic | istella | Xq | - | 7.0 | 6.9 | - | - | - | sklearn-cpu 2.3 ms (IDENTICAL/arm 3.051) |
| algos | isotonic | taxi | Xq | - | 9.3 | 9.1 | - | - | - | sklearn-cpu 4.6 ms (IDENTICAL/arm 1.989) |
| algos | knn-imputer | taxi | Xq | - | 338.2 | 317.4 | - | - | - | sklearn-cpu 25457.6 ms (IDENTICAL/arm 0.012) |
| algos | label-propagation | istella | Xq | - | - | - | - | - | - | sklearn-cpu 1268.4 ms (IDENTICAL/arm -) |
| algos | label-propagation | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 1004.6 ms (IDENTICAL/arm -) |
| algos | label-spreading | istella | Xq | - | - | - | - | - | - | sklearn-cpu 1277.1 ms (IDENTICAL/arm -) |
| algos | label-spreading | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 959.8 ms (IDENTICAL/arm -) |
| algos | lars | istella | Xq | - | 16.0 | 16.4 | - | - | - | sklearn-cpu - ms (IDENTICAL/arm -) |
| algos | lars | taxi | Xq | - | 1.0 | 1.1 | - | - | - | sklearn-cpu 0.3 ms (IDENTICAL/arm 3.320) |
| algos | lasso-cv | istella | Xq | - | 17.4 | 17.3 | - | - | - | sklearn-cpu 3.9 ms (IDENTICAL/arm 4.477) |
| algos | lasso-cv | taxi | Xq | - | 2.1 | 2.2 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 5.898) |
| algos | lasso-lars | istella | Xq | - | 15.0 | 15.0 | - | - | - | sklearn-cpu 3.9 ms (IDENTICAL/arm 3.873) |
| algos | lasso-lars | taxi | Xq | - | 1.2 | 1.3 | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm 2.962) |
| algos | logreg-cv | istella | Xq | - | 57.2 | - | - | - | - | sklearn-cpu 24.3 ms (IDENTICAL/arm -) |
| algos | meanshift | istella | Xq | - | 20.0 | 19.5 | - | - | - | sklearn-cpu 11.8 ms (IDENTICAL/arm 1.653) |
| algos | meanshift | taxi | Xq | - | 2.5 | 2.5 | - | - | - | sklearn-cpu 18.7 ms (IDENTICAL/arm 0.132) |
| algos | minibatch-kmeans | istella | Xq | - | 17.4 | 17.8 | - | - | - | sklearn-cpu 6.1 ms (IDENTICAL/arm 2.913) |
| algos | minibatch-kmeans | taxi | Xq | - | 1.9 | 2.4 | - | - | - | sklearn-cpu 1.5 ms (IDENTICAL/arm 1.550) |
| algos | nearest-centroid | istella | Xq | - | 40.9 | 39.8 | - | - | - | sklearn-cpu 169.4 ms (IDENTICAL/arm 0.235) |
| algos | nearest-centroid | taxi | Xq | - | 25.4 | 25.3 | - | - | - | sklearn-cpu 12.9 ms (IDENTICAL/arm 1.961) |
| algos | ocsvm | istella | Xq | - | 15.1 | 12.9 | - | - | - | sklearn-cpu 754.9 ms (IDENTICAL/arm 0.017) |
| algos | ocsvm | taxi | Xq | - | 4.5 | 7.1 | - | - | - | sklearn-cpu 222.0 ms (IDENTICAL/arm 0.032) |
| algos | optics | istella | Xq | - | 0.0 | 0.0 | - | - | - | sklearn-cpu 0.0 ms (IDENTICAL/arm 0.272) |
| algos | optics | taxi | Xq | - | 0.0 | 0.0 | - | - | - | sklearn-cpu 0.0 ms (IDENTICAL/arm 0.500) |
| algos | pa-clf | istella | Xq | - | 24.4 | 24.6 | - | - | - | sklearn-cpu 4.6 ms (IDENTICAL/arm 5.367) |
| algos | pa-clf | taxi | Xq | - | 16.4 | 15.9 | - | - | - | sklearn-cpu 1.0 ms (IDENTICAL/arm 16.187) |
| algos | pa-reg | istella | Xq | - | 22.2 | 21.2 | - | - | - | sklearn-cpu 4.1 ms (IDENTICAL/arm 5.175) |
| algos | pa-reg | taxi | Xq | - | 8.7 | 12.4 | - | - | - | sklearn-cpu 0.5 ms (IDENTICAL/arm 24.039) |
| algos | perceptron | istella | Xq | - | 24.5 | 23.6 | - | - | - | sklearn-cpu 4.4 ms (IDENTICAL/arm 5.303) |
| algos | perceptron | taxi | Xq | - | 16.0 | 15.9 | - | - | - | sklearn-cpu 1.0 ms (IDENTICAL/arm 15.861) |
| algos | poisson | taxi | Xq | - | 8.3 | 8.2 | - | - | - | sklearn-cpu 3.1 ms (IDENTICAL/arm 2.609) |
| algos | poly-count-sketch | istella | Xq | - | 2.1 | 2.2 | - | - | - | sklearn-cpu 4.9 ms (IDENTICAL/arm 0.454) |
| algos | poly-count-sketch | taxi | Xq | - | 1.7 | 1.9 | - | - | - | sklearn-cpu 2.7 ms (IDENTICAL/arm 0.702) |
| algos | quantile | istella | Xq | - | 14.7 | 18.5 | - | - | - | sklearn-cpu 16.2 ms (IDENTICAL/arm 1.140) |
| algos | quantile | taxi | Xq | - | 8.3 | 8.1 | - | - | - | sklearn-cpu 1.3 ms (IDENTICAL/arm 6.149) |
| algos | radius-neighbors | istella | Xq | - | 2991.3 | 3129.9 | - | - | - | sklearn-cpu 1216.1 ms (IDENTICAL/arm 2.574) |
| algos | radius-neighbors | taxi | Xq | - | 151.9 | 164.1 | - | - | - | sklearn-cpu 35.8 ms (IDENTICAL/arm 4.580) |
| algos | ridge-clf | istella | Xq | - | 22.4 | 21.9 | - | - | - | sklearn-cpu 4.4 ms (IDENTICAL/arm 4.986) |
| algos | ridge-clf | taxi | Xq | - | 9.3 | 9.7 | - | - | - | sklearn-cpu 0.8 ms (IDENTICAL/arm 12.014) |
| algos | ridge-cv | istella | Xq | - | - | - | - | - | - | sklearn-cpu 3.9 ms (IDENTICAL/arm -) |
| algos | ridge-cv | taxi | Xq | - | - | - | - | - | - | sklearn-cpu 0.4 ms (IDENTICAL/arm -) |
| algos | sgd-clf | istella | Xq | - | 23.8 | 23.2 | - | - | - | sklearn-cpu 4.6 ms (IDENTICAL/arm 5.091) |
| algos | sgd-clf | taxi | Xq | - | 16.1 | 14.1 | - | - | - | sklearn-cpu 0.9 ms (IDENTICAL/arm 14.893) |
| algos | sgd-ocsvm | istella | Xq | - | 22.6 | 23.0 | - | - | - | sklearn-cpu 4.3 ms (IDENTICAL/arm 5.397) |
| algos | sgd-ocsvm | taxi | Xq | - | 17.7 | 17.4 | - | - | - | sklearn-cpu 0.6 ms (IDENTICAL/arm 29.499) |
| algos | sgd-reg | istella | Xq | - | 20.4 | 21.5 | - | - | - | sklearn-cpu 4.1 ms (IDENTICAL/arm 5.171) |
| algos | sgd-reg | taxi | Xq | - | 11.9 | 8.8 | - | - | - | sklearn-cpu 0.6 ms (IDENTICAL/arm 15.334) |
| algos | skewed-chi2 | istella | Xq | - | 3.2 | 3.2 | - | - | - | sklearn-cpu 1.1 ms (IDENTICAL/arm 2.947) |
| algos | skewed-chi2 | taxi | Xq | - | 2.1 | 2.1 | - | - | - | sklearn-cpu 0.6 ms (IDENTICAL/arm 3.339) |
| classical | kmeans | istella | Xq | 500000 | 124.4 | 130.2 | yes | - | - | sklearn-cpu 57.5 ms (IDENTICAL/arm 2.263); torch-gpu 45.4 ms (IDENTICAL/arm 2.871) |
| classical | kmeans | taxi | Xq | 500000 | 36.7 | 39.3 | yes | - | - | sklearn-cpu 15.6 ms (IDENTICAL/arm 2.516); torch-gpu 38.7 ms (IDENTICAL/arm 1.014) |
| classical | ols | istella | Xq | 500000 | 84.3 | 72.6 | no | - | - | sklearn-cpu 51.5 ms (IDENTICAL/arm 1.408); torch-gpu - ms (IDENTICAL/arm -) |
| classical | ols | taxi | Xq | 500000 | 32.4 | 32.4 | no | - | - | sklearn-cpu 6.9 ms (IDENTICAL/arm 4.715); torch-gpu - ms (IDENTICAL/arm -) |
| classical | pca | istella | Xq | 500000 | 119.7 | 120.9 | no | - | - | sklearn-cpu 59.5 ms (IDENTICAL/arm 2.034); torch-gpu - ms (IDENTICAL/arm -) |
| classical | pca | taxi | Xq | 500000 | 43.6 | 45.4 | no | - | - | sklearn-cpu 22.8 ms (IDENTICAL/arm 1.994); torch-gpu - ms (IDENTICAL/arm -) |
| classical | svc | istella | Xq | 10000 | 75.3 | 46.4 | yes | - | - | sklearn-cpu 1831.7 ms (IDENTICAL/arm 0.025) |
| classical | svc | taxi | Xq | 10000 | 25.5 | 65.4 | yes | - | - | sklearn-cpu 1348.6 ms (IDENTICAL/arm 0.048) |
| trees | et | istella | test | 500000 | 42.7 | 48.1 | yes | - | - | sklearn-et-cpu 310.6 ms (IDENTICAL/arm 0.155); lightgbm-cpu 471.9 ms (IDENTICAL/arm 0.102) |
| trees | et | istella | large | 1000000 | 84.6 | 104.3 | yes | - | - | sklearn-et-cpu 493.5 ms (IDENTICAL/arm 0.211); lightgbm-cpu 927.8 ms (IDENTICAL/arm 0.112) |
| trees | et | taxi | test | 500000 | 16.6 | 19.4 | yes | - | - | sklearn-et-cpu 135.4 ms (IDENTICAL/arm 0.144); lightgbm-cpu 278.7 ms (IDENTICAL/arm 0.070) |
| trees | et | taxi | large | 1000000 | 31.9 | 32.4 | yes | - | - | sklearn-et-cpu 303.4 ms (IDENTICAL/arm 0.107); lightgbm-cpu 474.0 ms (IDENTICAL/arm 0.068) |
| trees | gbdt-categorical | taxi | test | 500000 | 378.4 | 402.9 | no | - | - | catboost-cpu 636.4 ms (IDENTICAL/arm 0.633); xgboost-cpu 276.8 ms (IDENTICAL/arm 1.455); lightgbm-cpu 765.4 ms (IDENTICAL/arm 0.526) |
| trees | gbdt-categorical | taxi | large | 1000000 | 477.2 | 486.2 | no | - | - | catboost-cpu 1279.8 ms (IDENTICAL/arm 0.380); xgboost-cpu 541.1 ms (IDENTICAL/arm 0.898); lightgbm-cpu 1562.4 ms (IDENTICAL/arm 0.311) |
| trees | gbdt-depthwise | istella | test | 500000 | 388.1 | 387.9 | no | - | - | catboost-cpu 185.3 ms (IDENTICAL/arm 2.094); xgboost-cpu 102.5 ms (IDENTICAL/arm 3.784) |
| trees | gbdt-depthwise | istella | large | 1000000 | 478.3 | 461.7 | no | - | - | catboost-cpu 364.7 ms (IDENTICAL/arm 1.266); xgboost-cpu 205.6 ms (IDENTICAL/arm 2.246) |
| trees | gbdt-depthwise | taxi | test | 500000 | 322.6 | 327.8 | no | - | - | catboost-cpu 137.4 ms (IDENTICAL/arm 2.386); xgboost-cpu 92.6 ms (IDENTICAL/arm 3.539) |
| trees | gbdt-depthwise | taxi | large | 1000000 | 389.9 | 374.7 | no | - | - | catboost-cpu 265.9 ms (IDENTICAL/arm 1.409); xgboost-cpu 185.4 ms (IDENTICAL/arm 2.021) |
| trees | gbdt-lossguide | istella | test | 500000 | 379.5 | 410.9 | no | - | - | catboost-cpu 187.1 ms (IDENTICAL/arm 2.196); xgboost-cpu 113.4 ms (IDENTICAL/arm 3.622); lightgbm-cpu 620.3 ms (IDENTICAL/arm 0.662) |
| trees | gbdt-lossguide | istella | large | 1000000 | 481.9 | 517.8 | no | - | - | catboost-cpu 370.3 ms (IDENTICAL/arm 1.398); xgboost-cpu 222.6 ms (IDENTICAL/arm 2.326); lightgbm-cpu 1238.3 ms (IDENTICAL/arm 0.418) |
| trees | gbdt-lossguide | taxi | test | 500000 | 351.0 | 361.1 | no | - | - | catboost-cpu 137.3 ms (IDENTICAL/arm 2.630); xgboost-cpu 96.2 ms (IDENTICAL/arm 3.754); lightgbm-cpu 726.1 ms (IDENTICAL/arm 0.497) |
| trees | gbdt-lossguide | taxi | large | 1000000 | 373.1 | 395.2 | no | - | - | catboost-cpu 267.9 ms (IDENTICAL/arm 1.475); xgboost-cpu 191.9 ms (IDENTICAL/arm 2.059); lightgbm-cpu 1290.6 ms (IDENTICAL/arm 0.306) |
| trees | gbdt-multiclass | istella | test | 500000 | 127.5 | 136.9 | no | - | - | catboost-cpu 87.3 ms (IDENTICAL/arm 1.567); xgboost-cpu 597.0 ms (IDENTICAL/arm 0.229); lightgbm-cpu 4064.8 ms (IDENTICAL/arm 0.034) |
| trees | gbdt-multiclass | istella | large | 1000000 | 224.2 | 238.3 | no | - | - | catboost-cpu 163.8 ms (IDENTICAL/arm 1.455); xgboost-cpu 1198.9 ms (IDENTICAL/arm 0.199); lightgbm-cpu 7987.5 ms (IDENTICAL/arm 0.030) |
| trees | gbdt-multiclass | taxi | test | 500000 | 62.3 | 64.6 | no | - | - | catboost-cpu 35.6 ms (IDENTICAL/arm 1.814); xgboost-cpu 472.9 ms (IDENTICAL/arm 0.137); lightgbm-cpu 3292.3 ms (IDENTICAL/arm 0.020) |
| trees | gbdt-multiclass | taxi | large | 1000000 | 104.1 | 104.3 | no | - | - | catboost-cpu 66.8 ms (IDENTICAL/arm 1.561); xgboost-cpu 1030.9 ms (IDENTICAL/arm 0.101); lightgbm-cpu 6501.7 ms (IDENTICAL/arm 0.016) |
| trees | gbdt-rank-pairlogit | istella | test | 681250 | 71.9 | 64.1 | no | - | - | catboost-cpu 54.0 ms (IDENTICAL/arm 1.187); xgboost-cpu 30.1 ms (IDENTICAL/arm 2.130) |
| trees | gbdt-rank-pairlogit | istella | large | 1000000 | 100.2 | 99.1 | no | - | - | catboost-cpu 77.0 ms (IDENTICAL/arm 1.286); xgboost-cpu 41.3 ms (IDENTICAL/arm 2.401) |
| trees | gbdt-rank-yetirank | istella | test | 681250 | 69.7 | 71.4 | no | - | - | catboost-cpu 53.2 ms (IDENTICAL/arm 1.343); xgboost-cpu 30.9 ms (IDENTICAL/arm 2.312); lightgbm-cpu 113.2 ms (IDENTICAL/arm 0.631) |
| trees | gbdt-rank-yetirank | istella | large | 1000000 | 99.1 | 102.3 | no | - | - | catboost-cpu 71.1 ms (IDENTICAL/arm 1.439); xgboost-cpu 40.4 ms (IDENTICAL/arm 2.536); lightgbm-cpu 163.3 ms (IDENTICAL/arm 0.627) |
| trees | gbdt-symmetric-1000 | istella | test | 500000 | 80.2 | 73.2 | no | - | - | catboost-cpu 70.3 ms (IDENTICAL/arm 1.042) |
| trees | gbdt-symmetric-1000 | istella | large | 1000000 | 136.1 | 124.2 | no | - | - | catboost-cpu 134.2 ms (IDENTICAL/arm 0.926) |
| trees | gbdt-symmetric-1000 | taxi | test | 500000 | 45.4 | 33.1 | no | - | - | catboost-cpu 34.1 ms (IDENTICAL/arm 0.970) |
| trees | gbdt-symmetric-1000 | taxi | large | 1000000 | 71.0 | 59.3 | no | - | - | catboost-cpu 60.4 ms (IDENTICAL/arm 0.981) |
| trees | gbdt-symmetric | istella | test | 500000 | 68.3 | 62.4 | no | - | - | catboost-cpu 57.3 ms (IDENTICAL/arm 1.090) |
| trees | gbdt-symmetric | istella | large | 1000000 | 113.2 | 114.2 | no | - | - | catboost-cpu 105.7 ms (IDENTICAL/arm 1.081) |
| trees | gbdt-symmetric | taxi | test | 500000 | 24.2 | 18.4 | no | - | - | catboost-cpu 24.2 ms (IDENTICAL/arm 0.761) |
| trees | gbdt-symmetric | taxi | large | 1000000 | 38.0 | 31.7 | no | - | - | catboost-cpu 38.2 ms (IDENTICAL/arm 0.830) |
| trees | iforest | istella | test | 500000 | 738.7 | 760.4 | no | - | - | sklearn-iforest-cpu 860.8 ms (IDENTICAL/arm 0.883) |
| trees | iforest | istella | large | 1000000 | 1138.2 | 1143.5 | no | - | - | sklearn-iforest-cpu 1692.3 ms (IDENTICAL/arm 0.676) |
| trees | iforest | taxi | test | 500000 | 136.6 | 137.4 | no | - | - | sklearn-iforest-cpu 842.0 ms (IDENTICAL/arm 0.163) |
| trees | iforest | taxi | large | 1000000 | 163.7 | 173.0 | no | - | - | sklearn-iforest-cpu 1656.7 ms (IDENTICAL/arm 0.104) |
| trees | rf | istella | test | 500000 | 81.0 | 119.8 | yes | - | - | sklearn-rf-cpu 990.6 ms (IDENTICAL/arm 0.121); lightgbm-cpu 662.9 ms (IDENTICAL/arm 0.181) |
| trees | rf | istella | large | 1000000 | 159.8 | 208.2 | yes | - | - | sklearn-rf-cpu 1906.1 ms (IDENTICAL/arm 0.109); lightgbm-cpu 1312.1 ms (IDENTICAL/arm 0.159) |
| trees | rf | taxi | test | 500000 | 52.7 | 60.8 | yes | - | - | sklearn-rf-cpu 390.4 ms (IDENTICAL/arm 0.156); lightgbm-cpu 612.2 ms (IDENTICAL/arm 0.099) |
| trees | rf | taxi | large | 1000000 | 105.0 | 123.2 | yes | - | - | sklearn-rf-cpu 774.6 ms (IDENTICAL/arm 0.159); lightgbm-cpu 1223.9 ms (IDENTICAL/arm 0.101) |

## Trees

### et / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/et.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4401.8 | 4401.8..4401.8 | 1 | - | - | - | 6813.7 | - | auc=0.937987, logloss=0.189989 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4360.2 | 4360.2..4360.2 | 1 | - | - | - | 8634.0 | - | auc=0.937987, logloss=0.189989 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 30616.6 | 30616.6..30616.6 | 1 | 0.144 | 0.142 | - | 5476.8 | - | auc=0.937904, logloss=0.190078 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 520331.3 | 520331.3..520331.3 | 1 | 0.008 | 0.008 | - | 25185.5 | - | auc=0.948128, logloss=0.197270 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:769496,ours:1029236,ours-ab:1029236,sklearn-et-cpu:999969 spread=0.2524 verdict=NOT-COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-ab | sklearn-et-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - | - |
| bootstrap | - | false | false | false |
| class_weight | null | null | null | null |
| criterion | - | "gini" | "gini" | "gini" |
| feature_fraction | 1.0 | - | - | - |
| feature_fraction_bynode | 0.06363636363636363 | - | - | - |
| learning_rate | 1.0 | - | - | - |
| max_bin | 255 | - | - | - |
| max_depth | 16 | 16 | 16 | 16 |
| max_features | - | "sqrt" | "sqrt" | "sqrt" |
| max_leaves | 65536 | null | null | null |
| max_samples | - | null | null | null |
| min_child_weight | 0.0 | - | - | - |
| min_samples_leaf | 1 | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| reg_alpha | 0.0 | - | - | - |
| reg_lambda | 0.0 | - | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | 0.632 | - | - | - |

accepted difference: ours-ab class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: ours-ab max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: ours-ab max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

accepted difference: sklearn-et-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-et-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: sklearn-et-cpu max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 48.1 | 48.1..48.1 | 1 | - | - | - | auc=0.937987, auc_matches_fit=True, logloss=0.189989, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 42.7 | 42.7..42.7 | 1 | - | - | - | auc=0.937987, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.189989, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 310.6 | 310.6..310.6 | 1 | 0.155 | 0.138 | - | auc=0.937904, auc_matches_fit=True, logloss=0.190078, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 471.9 | 471.9..471.9 | 1 | 0.102 | 0.091 | - | auc=0.948128, auc_matches_fit=True, logloss=0.197270, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 104.3 | 104.3..104.3 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 84.6 | 84.6..84.6 | 1 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 493.5 | 493.5..493.5 | 1 | 0.211 | 0.171 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 927.8 | 927.8..927.8 | 1 | 0.112 | 0.091 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### et / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/et.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3399.7 | 3399.7..3399.7 | 1 | - | - | - | 2492.0 | - | auc=0.618907, logloss=0.526142 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3376.0 | 3376.0..3376.0 | 1 | - | - | - | 2579.7 | - | auc=0.618907, logloss=0.526142 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 20628.7 | 20628.7..20628.7 | 1 | 0.165 | 0.164 | - | 3907.4 | - | auc=0.618972, logloss=0.525976 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 51349.2 | 51349.2..51349.2 | 1 | 0.066 | 0.066 | - | 2350.3 | - | auc=0.611172, logloss=0.529303 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:66043,ours:881399,ours-ab:881399,sklearn-et-cpu:916826 spread=0.9280 verdict=NOT-COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-ab | sklearn-et-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - | - |
| bootstrap | - | false | false | false |
| class_weight | null | null | null | null |
| criterion | - | "gini" | "gini" | "gini" |
| feature_fraction | 1.0 | - | - | - |
| feature_fraction_bynode | 0.25 | - | - | - |
| learning_rate | 1.0 | - | - | - |
| max_bin | 255 | - | - | - |
| max_depth | 16 | 16 | 16 | 16 |
| max_features | - | "sqrt" | "sqrt" | "sqrt" |
| max_leaves | 65536 | null | null | null |
| max_samples | - | null | null | null |
| min_child_weight | 0.0 | - | - | - |
| min_samples_leaf | 1 | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| reg_alpha | 0.0 | - | - | - |
| reg_lambda | 0.0 | - | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | 0.632 | - | - | - |

accepted difference: ours-ab class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: ours-ab max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: ours-ab max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

accepted difference: sklearn-et-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-et-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: sklearn-et-cpu max_samples: bootstrap False on ours and sklearn: every row in every tree; sklearn refuses max_samples without bootstrap, so it stays None on both

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours and sklearn max_leaf_nodes None, LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 19.4 | 19.4..19.4 | 1 | - | - | - | auc=0.618907, auc_matches_fit=True, logloss=0.526142, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 16.6 | 16.6..16.6 | 1 | - | - | - | auc=0.618907, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.526142, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | test | 500000 | 135.4 | 135.4..135.4 | 1 | 0.144 | 0.123 | - | auc=0.618972, auc_matches_fit=True, logloss=0.525976, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 278.7 | 278.7..278.7 | 1 | 0.070 | 0.060 | - | auc=0.611172, auc_matches_fit=True, logloss=0.529303, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 32.4 | 32.4..32.4 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 31.9 | 31.9..31.9 | 1 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-et-cpu | large | 1000000 | 303.4 | 303.4..303.4 | 1 | 0.107 | 0.105 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 474.0 | 474.0..474.0 | 1 | 0.068 | 0.067 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn ExtraTreesClassifier.predict_proba(X), column 1

inference call, sklearn-et-cpu: sklearn ExtraTreesClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### gbdt-categorical / taxi (rows full, shape taxicat-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-categorical.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 62463.8 | 62463.8..62463.8 | 1 | - | - | - | 6431.8 | - | auc=0.630363, logloss=0.528463 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 63398.2 | 63398.2..63398.2 | 1 | - | - | - | 6979.1 | - | auc=0.630212, logloss=0.528575 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 95141.8 | 95141.8..95141.8 | 1 | 0.657 | 0.666 | - | 9406.3 | - | auc=0.628462, logloss=0.529047 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 60862.8 | 60862.8..60862.8 | 1 | 1.026 | 1.042 | - | 7483.2 | - | auc=0.631472, logloss=0.528686 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 46900.2 | 46900.2..46900.2 | 1 | 1.332 | 1.352 | - | 7474.9 | - | auc=0.632665, logloss=0.528094 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-categorical arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:76180,lightgbm-cpu:92031,ours:42048,ours-ab:39749,xgboost-cpu:99610 spread=0.6010 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (binary task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | "No" | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "Lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 0.0 | - | 0.0 | 0.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | - | "NewtonL2" | "NewtonL2" | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | null | 1.0 |

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu score_function: CatBoost's CPU learner scores splits with Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and LightGBM

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 402.9 | 402.9..402.9 | 1 | - | - | - | auc=0.630363, auc_matches_fit=True, logloss=0.528463, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 378.4 | 378.4..378.4 | 1 | - | - | - | auc=0.630212, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.528575, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.540728 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 636.4 | 636.4..636.4 | 1 | 0.633 | 0.595 | - | auc=0.628462, auc_matches_fit=True, logloss=0.529047, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 276.8 | 276.8..276.8 | 1 | 1.455 | 1.367 | - | auc=0.631472, auc_matches_fit=True, logloss=0.528686, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 765.4 | 765.4..765.4 | 1 | 0.526 | 0.494 | - | auc=0.632665, auc_matches_fit=True, logloss=0.528094, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 486.2 | 486.2..486.2 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 477.2 | 477.2..477.2 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.504951 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 1279.8 | 1279.8..1279.8 | 1 | 0.380 | 0.373 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 541.1 | 541.1..541.1 | 1 | 0.898 | 0.882 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1562.4 | 1562.4..1562.4 | 1 | 0.311 | 0.305 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X) on the float32 codes, column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X) on the float32 codes, column 1

inference call, catboost-cpu: catboost predict_proba(int64 categorical frame built in the clock, task_type CPU), column 1

inference call, xgboost-cpu: xgboost XGBClassifier.predict_proba(pandas CategoricalDtype frame built in the clock; inplace_predict takes no category frame here), column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) on the float32 codes (its categorical columns are recorded in the model), probability

### gbdt-depthwise / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16013.0 | 16013.0..16013.0 | 1 | - | - | - | 8925.6 | - | auc=0.979129, logloss=0.188483 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19061.9 | 19061.9..19061.9 | 1 | - | - | - | 9849.8 | - | auc=0.980182, logloss=0.181879 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 85010.0 | 85010.0..85010.0 | 1 | 0.188 | 0.224 | - | 8824.9 | - | auc=0.983136, logloss=0.157685 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 21590.0 | 21590.0..21590.0 | 1 | 0.742 | 0.883 | - | 9074.0 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:105455,ours:38550,ours-ab:44863,xgboost-cpu:107149 spread=0.6402 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | "No" | "No" | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "Depthwise" | "Depthwise" | "Depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | 1 | - |
| leaf_estimation_method | "Newton" | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | null | null | 0.0 |
| min_samples_leaf | 1 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | "Min" | - |
| random_strength | 0.0 | 0.0 | 0.0 | - |
| reg_alpha | - | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 387.9 | 387.9..387.9 | 1 | - | - | - | auc=0.979129, auc_matches_fit=True, logloss=0.188483, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 388.1 | 388.1..388.1 | 1 | - | - | - | auc=0.980182, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.181879, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.469088 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 185.3 | 185.3..185.3 | 1 | 2.094 | 2.095 | - | auc=0.983136, auc_matches_fit=True, logloss=0.157685, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 102.5 | 102.5..102.5 | 1 | 3.784 | 3.786 | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 461.7 | 461.7..461.7 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 478.3 | 478.3..478.3 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.520389 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 364.7 | 364.7..364.7 | 1 | 1.266 | 1.311 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 205.6 | 205.6..205.6 | 1 | 2.246 | 2.326 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-depthwise / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 16151.6 | 16151.6..16151.6 | 1 | - | - | - | 3654.4 | - | auc=0.625417, logloss=0.530232 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13994.4 | 13994.4..13994.4 | 1 | - | - | - | 3725.9 | - | auc=0.625840, logloss=0.530036 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 52858.3 | 52858.3..52858.3 | 1 | 0.306 | 0.265 | - | 3916.3 | - | auc=0.632459, logloss=0.527789 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 9923.3 | 9923.3..9923.3 | 1 | 1.628 | 1.410 | - | 3952.4 | - | auc=0.630969, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:103111,ours:41324,ours-ab:29604,xgboost-cpu:91234 spread=0.7129 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | "No" | "No" | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "Depthwise" | "Depthwise" | "Depthwise" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | 1 | - |
| leaf_estimation_method | "Newton" | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 |
| min_child_weight | - | null | null | 0.0 |
| min_samples_leaf | 1 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | "Min" | - |
| random_strength | 0.0 | 0.0 | 0.0 | - |
| reg_alpha | - | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 327.8 | 327.8..327.8 | 1 | - | - | - | auc=0.625417, auc_matches_fit=True, logloss=0.530232, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 322.6 | 322.6..322.6 | 1 | - | - | - | auc=0.625840, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.530036, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.768170 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 137.4 | 137.4..137.4 | 1 | 2.386 | 2.348 | - | auc=0.632459, auc_matches_fit=True, logloss=0.527789, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 92.6 | 92.6..92.6 | 1 | 3.539 | 3.484 | - | auc=0.630969, auc_matches_fit=True, logloss=0.528677, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 374.7 | 374.7..374.7 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 389.9 | 389.9..389.9 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.680839 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 265.9 | 265.9..265.9 | 1 | 1.409 | 1.466 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 185.4 | 185.4..185.4 | 1 | 2.021 | 2.103 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

### gbdt-lossguide / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 97126.4 | 97126.4..97126.4 | 1 | - | - | - | 8320.6 | - | auc=0.983749, logloss=0.149364 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 106296.1 | 106296.1..106296.1 | 1 | - | - | - | 9246.8 | - | auc=0.983790, logloss=0.148810 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 105604.4 | 105604.4..105604.4 | 1 | 0.920 | 1.007 | - | 8401.1 | - | auc=0.983136, logloss=0.157685 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 72236.9 | 72236.9..72236.9 | 1 | 1.345 | 1.471 | - | 8433.8 | - | auc=0.983622, logloss=0.149263 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 54183.0 | 54183.0..54183.0 | 1 | 1.793 | 1.962 | - | 8772.8 | - | auc=0.983778, logloss=0.149653 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:105455,lightgbm-cpu:94845,ours:113893,ours-ab:112092,xgboost-cpu:107149 spread=0.1672 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | "No" | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "Lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 0.0 | - | 0.0 | 0.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | - | "NewtonL2" | "NewtonL2" | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | null | 1.0 |

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu score_function: CatBoost's CPU learner scores splits with Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and LightGBM

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 410.9 | 410.9..410.9 | 1 | - | - | - | auc=0.983749, auc_matches_fit=True, logloss=0.149364, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 379.5 | 379.5..379.5 | 1 | - | - | - | auc=0.983790, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.148810, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.643391 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 187.1 | 187.1..187.1 | 1 | 2.196 | 2.028 | - | auc=0.983136, auc_matches_fit=True, logloss=0.157685, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 113.4 | 113.4..113.4 | 1 | 3.622 | 3.345 | - | auc=0.983622, auc_matches_fit=True, logloss=0.149263, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 620.3 | 620.3..620.3 | 1 | 0.662 | 0.612 | - | auc=0.983778, auc_matches_fit=True, logloss=0.149653, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 517.8 | 517.8..517.8 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 481.9 | 481.9..481.9 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.515789 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 370.3 | 370.3..370.3 | 1 | 1.398 | 1.301 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 222.6 | 222.6..222.6 | 1 | 2.326 | 2.165 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1238.3 | 1238.3..1238.3 | 1 | 0.418 | 0.389 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### gbdt-lossguide / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 47377.7 | 47377.7..47377.7 | 1 | - | - | - | 3044.4 | - | auc=0.631154, logloss=0.528317 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 44556.5 | 44556.5..44556.5 | 1 | - | - | - | 3124.0 | - | auc=0.631046, logloss=0.528287 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 71543.8 | 71543.8..71543.8 | 1 | 0.662 | 0.623 | - | 3282.2 | - | auc=0.632459, logloss=0.527789 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 52765.0 | 52765.0..52765.0 | 1 | 0.898 | 0.844 | - | 3075.8 | - | auc=0.630969, logloss=0.528677 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 41349.6 | 41349.6..41349.6 | 1 | 1.146 | 1.078 | - | 2978.9 | - | auc=0.632243, logloss=0.528067 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:103111,lightgbm-cpu:79876,ours:56739,ours-ab:46329,xgboost-cpu:91234 spread=0.5507 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | "No" | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | - | 1.0 |
| grow_policy | "Lossguide" | - | "Lossguide" | "Lossguide" | "lossguide" |
| leaf_estimation_iterations | 1 | - | 1 | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | - | "Logloss" | "Logloss" | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 0.0 | - | 0.0 | 0.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | - | "NewtonL2" | "NewtonL2" | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | null | 1.0 |

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu score_function: CatBoost's CPU learner scores splits with Cosine or L2 only; ours and catboost-gpu NewtonL2, the Newton L2 gain of XGBoost and LightGBM

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 361.1 | 361.1..361.1 | 1 | - | - | - | auc=0.631154, auc_matches_fit=True, logloss=0.528317, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 351.0 | 351.0..351.0 | 1 | - | - | - | auc=0.631046, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.528287, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.524101 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 137.3 | 137.3..137.3 | 1 | 2.630 | 2.557 | - | auc=0.632459, auc_matches_fit=True, logloss=0.527789, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 96.2 | 96.2..96.2 | 1 | 3.754 | 3.650 | - | auc=0.630969, auc_matches_fit=True, logloss=0.528677, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 726.1 | 726.1..726.1 | 1 | 0.497 | 0.483 | - | auc=0.632243, auc_matches_fit=True, logloss=0.528067, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 395.2 | 395.2..395.2 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 373.1 | 373.1..373.1 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.493518 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 267.9 | 267.9..267.9 | 1 | 1.475 | 1.392 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 191.9 | 191.9..191.9 | 1 | 2.059 | 1.944 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1290.6 | 1290.6..1290.6 | 1 | 0.306 | 0.289 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix; XGBoost's documented fastest path), probability

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### gbdt-multiclass / istella (rows full, shape istellamc-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22808.1 | 22808.1..22808.1 | 1 | - | - | - | 9627.9 | - | accuracy=0.903294, mlogloss=0.281958 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21753.6 | 21753.6..21753.6 | 1 | - | - | - | 10506.9 | - | accuracy=0.903310, mlogloss=0.280935 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 736397.0 | 736397.0..736397.0 | 1 | 0.031 | 0.030 | - | 9490.4 | - | accuracy=0.907768, mlogloss=0.258286 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 100165.7 | 100165.7..100165.7 | 1 | 0.228 | 0.217 | - | 9820.7 | - | accuracy=0.910140, mlogloss=0.246803 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 220452.1 | 220452.1..220452.1 | 1 | 0.103 | 0.099 | - | 10034.5 | - | accuracy=0.910058, mlogloss=0.245916 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:128000,lightgbm-cpu:76782,ours:128000,ours-ab:128000,xgboost-cpu:88137 spread=0.4001 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | "No" | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "MultiClass" | - | "MultiClass" | "MultiClass" | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | null | null | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | null | null | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 1.0 | - | 1.0 | 1.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 136.9 | 136.9..136.9 | 1 | - | - | - | accuracy=0.903294, accuracy_matches_fit=True, mlogloss=0.281958, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 127.5 | 127.5..127.5 | 1 | - | - | - | accuracy=0.903310, accuracy_matches_fit=True, bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.296178, mlogloss=0.280935, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 87.3 | 87.3..87.3 | 1 | 1.567 | 1.460 | - | accuracy=0.907768, accuracy_matches_fit=True, mlogloss=0.258286, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 597.0 | 597.0..597.0 | 1 | 0.229 | 0.214 | - | accuracy=0.910140, accuracy_matches_fit=True, mlogloss=0.246803, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 4064.8 | 4064.8..4064.8 | 1 | 0.034 | 0.031 | - | accuracy=0.910058, accuracy_matches_fit=True, mlogloss=0.245916, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 238.3 | 238.3..238.3 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 224.2 | 224.2..224.2 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.400457 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 163.8 | 163.8..163.8 | 1 | 1.455 | 1.369 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 1198.9 | 1198.9..1198.9 | 1 | 0.199 | 0.187 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 7987.5 | 7987.5..7987.5 | 1 | 0.030 | 0.028 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU), the (rows, n_classes) probability matrix

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), the (rows, n_classes) probability matrix

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (on the CPU whatever device trained it), the (rows, n_classes) probability matrix

### gbdt-multiclass / taxi (rows full, shape taximc-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17555.9 | 17555.9..17555.9 | 1 | - | - | - | 4194.9 | - | accuracy=0.596646, mlogloss=1.022664 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 14684.9 | 14684.9..14684.9 | 1 | - | - | - | 4193.2 | - | accuracy=0.596646, mlogloss=1.022664 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 231474.7 | 231474.7..231474.7 | 1 | 0.076 | 0.063 | - | 4749.1 | - | accuracy=0.599150, mlogloss=1.012734 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 43845.9 | 43845.9..43845.9 | 1 | 0.400 | 0.335 | - | 4528.5 | - | accuracy=0.601128, mlogloss=1.005128 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 182718.0 | 182718.0..182718.0 | 1 | 0.096 | 0.080 | - | 4515.5 | - | accuracy=0.601580, mlogloss=1.004282 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:128000,lightgbm-cpu:90172,ours:128000,ours-ab:128000,xgboost-cpu:99745 spread=0.2955 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (MULTICLASS task) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | "No" | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "MultiClass" | - | "MultiClass" | "MultiClass" | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 | 256 | 256 |
| min_child_weight | - | 0.001 | null | null | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | null | null | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 | 500 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 1.0 | - | 1.0 | 1.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 64.6 | 64.6..64.6 | 1 | - | - | - | accuracy=0.596646, accuracy_matches_fit=True, mlogloss=1.022664, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 62.3 | 62.3..62.3 | 1 | - | - | - | accuracy=0.596646, accuracy_matches_fit=True, bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.364e-07, mlogloss=1.022664, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 35.6 | 35.6..35.6 | 1 | 1.814 | 1.751 | - | accuracy=0.599150, accuracy_matches_fit=True, mlogloss=1.012734, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 500000 | 472.9 | 472.9..472.9 | 1 | 0.137 | 0.132 | - | accuracy=0.601128, accuracy_matches_fit=True, mlogloss=1.005128, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 3292.3 | 3292.3..3292.3 | 1 | 0.020 | 0.019 | - | accuracy=0.601580, accuracy_matches_fit=True, mlogloss=1.004282, mlogloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 104.3 | 104.3..104.3 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 104.1 | 104.1..104.1 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.662e-07 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 66.8 | 66.8..66.8 | 1 | 1.561 | 1.559 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 1030.9 | 1030.9..1030.9 | 1 | 0.101 | 0.101 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 6501.7 | 6501.7..6501.7 | 1 | 0.016 | 0.016 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), the (rows, n_classes) probability matrix

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU), the (rows, n_classes) probability matrix

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), the (rows, n_classes) probability matrix

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (on the CPU whatever device trained it), the (rows, n_classes) probability matrix

### gbdt-rank-pairlogit / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-pairlogit.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4137.4 | 4137.4..4137.4 | 1 | - | - | - | 11075.0 | - | map=0.844605, ndcg10=0.711712, ndcg5=0.641863 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3996.7 | 3996.7..3996.7 | 1 | - | - | - | 12313.8 | - | map=0.841358, ndcg10=0.709249, ndcg5=0.639753 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 39048.2 | 39048.2..39048.2 | 1 | 0.106 | 0.102 | - | 11734.1 | - | map=0.846328, ndcg10=0.713361, ndcg5=0.643611 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 7482.1 | 7482.1..7482.1 | 1 | 0.553 | 0.534 | - | 12001.5 | - | map=0.872796, ndcg10=0.738397, ndcg5=0.670093 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-pairlogit arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6400,ours:4332,ours-ab:3990,xgboost-cpu:6395 spread=0.3766 verdict=NOT-COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | "No" | "No" | - |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | 1 | 1 | - |
| leaf_estimation_method | "Newton" | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "PairLogit" | "PairLogit" | "PairLogit" | - |
| max_bin | 255 | 255 | 255 | 255 |
| max_depth | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 |
| min_child_weight | - | null | null | 0.0 |
| min_samples_leaf | 1 | 1 | 1 | - |
| min_split_gain | - | null | null | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | "Min" | "Min" | - |
| random_strength | 1.0 | 1.0 | 1.0 | - |
| reg_alpha | - | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 681250 | 64.1 | 64.1..64.1 | 1 | - | - | - | map=0.844605, map_matches_fit=True, ndcg10=0.711712, ndcg10_matches_fit=True, ndcg5=0.641863, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 681250 | 71.9 | 71.9..71.9 | 1 | - | - | - | bits_equal_vs_ours_identical=False, map=0.841358, map_matches_fit=True, max_abs_diff_vs_ours_identical=0.699485, ndcg10=0.709249, ndcg10_matches_fit=True, ndcg5=0.639753, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 681250 | 54.0 | 54.0..54.0 | 1 | 1.187 | 1.333 | - | map=0.846328, map_matches_fit=True, ndcg10=0.713361, ndcg10_matches_fit=True, ndcg5=0.643611, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | test | 681250 | 30.1 | 30.1..30.1 | 1 | 2.130 | 2.392 | - | map=0.872796, map_matches_fit=True, ndcg10=0.738397, ndcg10_matches_fit=True, ndcg5=0.670093, ndcg5_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 99.1 | 99.1..99.1 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 100.2 | 100.2..100.2 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.699445 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 77.0 | 77.0..77.0 | 1 | 1.286 | 1.301 | - | - | yes | NOT-COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 41.3 | 41.3..41.3 | 1 | 2.401 | 2.428 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, ours-ab: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, catboost-cpu: catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the ranker: a host apply on every box), raw ranking scores

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), raw ranking scores

### gbdt-rank-yetirank / istella (rows full, shape istellarank-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-yetirank.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22540.2 | 22540.2..22540.2 | 1 | - | - | - | 8672.6 | - | map=0.814902, ndcg10=0.680993, ndcg5=0.615076 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 25773.2 | 25773.2..25773.2 | 1 | - | - | - | 9903.1 | - | map=0.814902, ndcg10=0.680982, ndcg5=0.615063 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 25296.0 | 25296.0..25296.0 | 1 | 0.891 | 1.019 | - | 9510.1 | - | map=0.851990, ndcg10=0.726111, ndcg5=0.660263 | yes | COMPARABLE | - | ok (measured this run) |
| xgboost-cpu | xgboost | cpu | opponent | 7763.1 | 7763.1..7763.1 | 1 | 2.904 | 3.320 | - | 9454.0 | - | map=0.842929, ndcg10=0.726256, ndcg5=0.664249 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 6764.7 | 6764.7..6764.7 | 1 | 3.332 | 3.810 | - | 9735.0 | - | map=0.858362, ndcg10=0.741515, ndcg5=0.680254 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-yetirank arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6400,lightgbm-cpu:6366,ours:6400,ours-ab:6400,xgboost-cpu:6400 spread=0.0053 verdict=COMPARABLE`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | lightgbm-cpu | ours | ours-ab | xgboost-cpu |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | catboost (get_params) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | xgboost (get_params) |
| boosting_type | "Plain" | "gbdt" | "Plain" | "Plain" | "gbtree" |
| bootstrap_type | "No" | - | "No" | "No" | - |
| class_weight | - | null | - | - | - |
| feature_border_type | "GreedyLogSum" | - | "GreedyLogSum" | "GreedyLogSum" | - |
| feature_fraction | - | 1.0 | 1.0 | 1.0 | 1.0 |
| feature_fraction_bynode | - | - | - | - | 1.0 |
| grow_policy | "SymmetricTree" | - | "SymmetricTree" | "SymmetricTree" | "depthwise" |
| leaf_estimation_iterations | 1 | - | 1 | 1 | - |
| leaf_estimation_method | "Newton" | - | "Newton" | "Newton" | - |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| loss | "YetiRank" | - | "YetiRank" | "YetiRank" | - |
| max_bin | 255 | 255 | 255 | 255 | 255 |
| max_depth | 6 | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 | 64 |
| min_child_weight | - | 0.001 | null | null | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | null | null | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 1.0 | - | 1.0 | 1.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | - | - | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | - | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 | 7 |
| subsample | null | 1.0 | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu grow_policy: ours and CatBoost fit this loss on the symmetric grower only; XGBoost has none and runs depthwise at the same depth

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: xgboost-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: lightgbm-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), LightGBM 'gbdt'; both plain gradient boosting

accepted difference: lightgbm-cpu min_child_weight: LightGBM 4.7.0 aborts a boosted tree at min_child_weight 0 (best_split_info.left_count > 0) and keeps 1e-3; ours has no hessian floor

accepted difference: lightgbm-cpu min_samples_leaf: LightGBM keeps min_child_samples 20 (it aborts at the other arms' value); ours and CatBoost min_data_in_leaf 1

accepted difference: lightgbm-cpu min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: lightgbm-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 681250 | 71.4 | 71.4..71.4 | 1 | - | - | - | map=0.814902, map_matches_fit=True, ndcg10=0.680993, ndcg10_matches_fit=True, ndcg5=0.615076, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 681250 | 69.7 | 69.7..69.7 | 1 | - | - | - | bits_equal_vs_ours_identical=False, map=0.814902, map_matches_fit=True, max_abs_diff_vs_ours_identical=0.0001413, ndcg10=0.680982, ndcg10_matches_fit=True, ndcg5=0.615063, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| catboost-cpu | test | 681250 | 53.2 | 53.2..53.2 | 1 | 1.343 | 1.311 | - | map=0.851990, map_matches_fit=True, ndcg10=0.726111, ndcg10_matches_fit=True, ndcg5=0.660263, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| xgboost-cpu | test | 681250 | 30.9 | 30.9..30.9 | 1 | 2.312 | 2.256 | - | map=0.842929, map_matches_fit=True, ndcg10=0.726256, ndcg10_matches_fit=True, ndcg5=0.664249, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cpu | test | 681250 | 113.2 | 113.2..113.2 | 1 | 0.631 | 0.616 | - | map=0.858362, map_matches_fit=True, ndcg10=0.741515, ndcg10_matches_fit=True, ndcg5=0.680254, ndcg5_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 102.3 | 102.3..102.3 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 99.1 | 99.1..99.1 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.0001436 | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 71.1 | 71.1..71.1 | 1 | 1.439 | 1.393 | - | - | yes | COMPARABLE | ok |
| xgboost-cpu | large | 1000000 | 40.4 | 40.4..40.4 | 1 | 2.536 | 2.455 | - | - | yes | COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 163.3 | 163.3..163.3 | 1 | 0.627 | 0.607 | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, ours-ab: mojolearn GradientBoosting.predict(X), raw ranking scores

inference call, catboost-cpu: catboost CatBoostRanker.predict(X, thread_count -1) (no task_type on the ranker: a host apply on every box), raw ranking scores

inference call, xgboost-cpu: xgboost Booster.inplace_predict(X) (no DMatrix), raw ranking scores

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (on the CPU whatever device trained it), raw ranking scores

### gbdt-symmetric-1000 / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 23713.3 | 23713.3..23713.3 | 1 | - | - | - | 7857.6 | - | auc=0.977075, logloss=0.203751 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21363.6 | 21363.6..21363.6 | 1 | - | - | - | 8775.2 | - | auc=0.975635, logloss=0.211169 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 114370.5 | 114370.5..114370.5 | 1 | 0.207 | 0.187 | - | 7981.4 | - | auc=0.982309, logloss=0.171620 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:256000,ours:67958,ours-ab:53524 spread=0.7909 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, 1000 trees (CatBoost's own default iteration count; see CATBOOST_DEFAULTS) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | "Logloss" |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | - | null | null |
| n_estimators | 1000 | 1000 | 1000 |
| nan_mode | "Min" | "Min" | "Min" |
| random_strength | 1.0 | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 73.2 | 73.2..73.2 | 1 | - | - | - | auc=0.977075, auc_matches_fit=True, logloss=0.203751, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 80.2 | 80.2..80.2 | 1 | - | - | - | auc=0.975635, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.211169, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.385475 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 70.3 | 70.3..70.3 | 1 | 1.042 | 1.141 | - | auc=0.982309, auc_matches_fit=True, logloss=0.171620, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 124.2 | 124.2..124.2 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 136.1 | 136.1..136.1 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.411415 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 134.2 | 134.2..134.2 | 1 | 0.926 | 1.015 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-symmetric-1000 / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21501.6 | 21501.6..21501.6 | 1 | - | - | - | 2504.6 | - | auc=0.621310, logloss=0.531329 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19085.6 | 19085.6..19085.6 | 1 | - | - | - | 2572.8 | - | auc=0.623944, logloss=0.530645 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 55516.0 | 55516.0..55516.0 | 1 | 0.387 | 0.344 | - | 2847.1 | - | auc=0.631642, logloss=0.528267 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:255568,ours:251900,ours-ab:232988 spread=0.0884 verdict=COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, 1000 trees (CatBoost's own default iteration count; see CATBOOST_DEFAULTS) (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | "Logloss" |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | - | null | null |
| n_estimators | 1000 | 1000 | 1000 |
| nan_mode | "Min" | "Min" | "Min" |
| random_strength | 1.0 | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 33.1 | 33.1..33.1 | 1 | - | - | - | auc=0.621310, auc_matches_fit=True, logloss=0.531329, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 45.4 | 45.4..45.4 | 1 | - | - | - | auc=0.623944, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.530645, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.149792 | yes | COMPARABLE | ok |
| catboost-cpu | test | 500000 | 34.1 | 34.1..34.1 | 1 | 0.970 | 1.329 | - | auc=0.631642, auc_matches_fit=True, logloss=0.528267, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 59.3 | 59.3..59.3 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 71.0 | 71.0..71.0 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.167560 | yes | COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 60.4 | 60.4..60.4 | 1 | 0.981 | 1.176 | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-symmetric / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13431.6 | 13431.6..13431.6 | 1 | - | - | - | 7842.1 | - | auc=0.977075, logloss=0.203751 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 11943.3 | 11943.3..11943.3 | 1 | - | - | - | 8752.7 | - | auc=0.975635, logloss=0.211169 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 58002.4 | 58002.4..58002.4 | 1 | 0.232 | 0.206 | - | 7970.1 | - | auc=0.979899, logloss=0.188093 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:128000,ours:66958,ours-ab:52524 spread=0.5897 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | "Logloss" |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | - | null | null |
| n_estimators | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | "Min" |
| random_strength | 1.0 | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 8.85789592328634 | 8.85789592328634 | 8.85789592328634 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 62.4 | 62.4..62.4 | 1 | - | - | - | auc=0.977075, auc_matches_fit=True, logloss=0.203751, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 68.3 | 68.3..68.3 | 1 | - | - | - | auc=0.975635, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.211169, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.385475 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 57.3 | 57.3..57.3 | 1 | 1.090 | 1.193 | - | auc=0.979899, auc_matches_fit=True, logloss=0.188093, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 114.2 | 114.2..114.2 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 113.2 | 113.2..113.2 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.411415 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 105.7 | 105.7..105.7 | 1 | 1.081 | 1.071 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### gbdt-symmetric / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10910.0 | 10910.0..10910.0 | 1 | - | - | - | 2551.2 | - | auc=0.621310, logloss=0.531329 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9520.0 | 9520.0..9520.0 | 1 | - | - | - | 2564.0 | - | auc=0.623944, logloss=0.530645 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 28176.7 | 28176.7..28176.7 | 1 | 0.387 | 0.338 | - | 2806.2 | - | auc=0.630269, logloss=0.528650 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:127568,ours:123900,ours-ab:104988 spread=0.1770 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, xgb/lgbm/cat shared_params, ntrees 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) |
| boosting_type | "Plain" | "Plain" | "Plain" |
| bootstrap_type | "No" | "No" | "No" |
| feature_border_type | "GreedyLogSum" | "GreedyLogSum" | "GreedyLogSum" |
| feature_fraction | - | 1.0 | 1.0 |
| grow_policy | "SymmetricTree" | "SymmetricTree" | "SymmetricTree" |
| leaf_estimation_iterations | 1 | 1 | 1 |
| leaf_estimation_method | "Newton" | "Newton" | "Newton" |
| learning_rate | 0.1 | 0.1 | 0.1 |
| loss | "Logloss" | "Logloss" | "Logloss" |
| max_bin | 255 | 255 | 255 |
| max_depth | 8 | 8 | 8 |
| max_leaves | 256 | 256 | 256 |
| min_child_weight | - | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | - | null | null |
| n_estimators | 500 | 500 | 500 |
| nan_mode | "Min" | "Min" | "Min" |
| random_strength | 1.0 | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| scale_pos_weight | 1.3101271632087197 | 1.3101271632087197 | 1.3101271632087197 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 18.4 | 18.4..18.4 | 1 | - | - | - | auc=0.621310, auc_matches_fit=True, logloss=0.531329, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 24.2 | 24.2..24.2 | 1 | - | - | - | auc=0.623944, auc_matches_fit=True, bits_equal_vs_ours_identical=False, logloss=0.530645, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.149790 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | test | 500000 | 24.2 | 24.2..24.2 | 1 | 0.761 | 0.999 | - | auc=0.630269, auc_matches_fit=True, logloss=0.528650, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 31.7 | 31.7..31.7 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 38.0 | 38.0..38.0 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=0.167560 | yes | NOT-COMPARABLE | ok |
| catboost-cpu | large | 1000000 | 38.2 | 38.2..38.2 | 1 | 0.830 | 0.993 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, ours-ab: mojolearn GradientBoosting.predict_proba(X), column 1

inference call, catboost-cpu: catboost predict_proba(X, task_type CPU, thread_count -1), column 1

### iforest / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/iforest.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 447.0 | 447.0..447.0 | 1 | - | - | - | 6244.4 | - | auc=0.830358 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 436.8 | 436.8..436.8 | 1 | - | - | - | 7182.6 | - | auc=0.830358 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 281.9 | 281.9..281.9 | 1 | 1.586 | 1.550 | - | 3792.1 | - | auc=0.827914 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:4534 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-ab | sklearn-iforest-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | false | false | false |
| contamination | "auto" | "auto" | "auto" |
| max_depth | null | null | - |
| max_features | 1.0 | 1.0 | 1.0 |
| max_samples | 256 | 256 | 256 |
| n_estimators | 100 | 100 | 100 |
| seed | 7 | 7 | 7 |

accepted difference: ours-ab max_depth: ours None is the auto depth ceil(log2(max_samples)) = 8; sklearn has no max_depth parameter and fixes the same value

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 760.4 | 760.4..760.4 | 1 | - | - | - | auc=0.830358, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn FAST | test | 500000 | 738.7 | 738.7..738.7 | 1 | - | - | - | auc=0.830358, auc_matches_fit=True, bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.96e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | test | 500000 | 860.8 | 860.8..860.8 | 1 | 0.883 | 0.858 | - | auc=0.827914, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 1143.5 | 1143.5..1143.5 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| mojolearn FAST | large | 1000000 | 1138.2 | 1138.2..1138.2 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=8.941e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | large | 1000000 | 1692.3 | 1692.3..1692.3 | 1 | 0.676 | 0.673 | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, ours-ab: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, sklearn-iforest-cpu: sklearn IsolationForest.score_samples(X), n_jobs -1

### iforest / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/iforest.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 123.2 | 123.2..123.2 | 1 | - | - | - | 1351.7 | - | auc=0.551846 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 118.9 | 118.9..118.9 | 1 | - | - | - | 1414.1 | - | auc=0.551846 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 464.5 | 464.5..464.5 | 1 | 0.265 | 0.256 | - | 949.8 | - | auc=0.552849 | yes | UNKNOWN | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:6131 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-ab | sklearn-iforest-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bootstrap | false | false | false |
| contamination | "auto" | "auto" | "auto" |
| max_depth | null | null | - |
| max_features | 1.0 | 1.0 | 1.0 |
| max_samples | 256 | 256 | 256 |
| n_estimators | 100 | 100 | 100 |
| seed | 7 | 7 | 7 |

accepted difference: ours-ab max_depth: ours None is the auto depth ceil(log2(max_samples)) = 8; sklearn has no max_depth parameter and fixes the same value

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 137.4 | 137.4..137.4 | 1 | - | - | - | auc=0.551846, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn FAST | test | 500000 | 136.6 | 136.6..136.6 | 1 | - | - | - | auc=0.551846, auc_matches_fit=True, bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.96e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | test | 500000 | 842.0 | 842.0..842.0 | 1 | 0.163 | 0.162 | - | auc=0.552849, auc_matches_fit=True | yes | UNKNOWN | ok |
| mojolearn IDENTICAL | large | 1000000 | 173.0 | 173.0..173.0 | 1 | - | - | - | - | yes | UNKNOWN | ok |
| mojolearn FAST | large | 1000000 | 163.7 | 163.7..163.7 | 1 | - | - | - | bits_equal_vs_ours_identical=False, max_abs_diff_vs_ours_identical=5.96e-08 | yes | UNKNOWN | ok |
| sklearn-iforest-cpu | large | 1000000 | 1656.7 | 1656.7..1656.7 | 1 | 0.104 | 0.099 | - | - | yes | UNKNOWN | ok |

inference call, ours: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, ours-ab: mojolearn IsolationForest.score_samples(X) (the forest is rebuilt inside every scoring call, DEVIATION 874, so this clock includes a forest build)

inference call, sklearn-iforest-cpu: sklearn IsolationForest.score_samples(X), n_jobs -1

### rf / istella (rows full, shape istella-2043304x220)

race: done, driver rc 0, log `raw/trees/rf.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14020.0 | 14020.0..14020.0 | 1 | - | - | - | 10034.9 | - | auc=0.945385, logloss=0.182017 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13810.0 | 13810.0..13810.0 | 1 | - | - | - | 10481.3 | - | auc=0.945385, logloss=0.182017 | yes | COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 92148.6 | 92148.6..92148.6 | 1 | 0.152 | 0.150 | - | 9316.0 | - | auc=0.945284, logloss=0.182344 | yes | COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 67209.9 | 67209.9..67209.9 | 1 | 0.209 | 0.205 | - | 7375.2 | - | auc=0.945361, logloss=0.195422 | yes | COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:125860,ours:125692,ours-ab:125692,sklearn-rf-cpu:125266 spread=0.0047 verdict=COMPARABLE`

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-ab | sklearn-rf-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - | - |
| bootstrap | - | true | true | true |
| class_weight | null | null | null | null |
| criterion | - | "gini" | "gini" | "gini" |
| feature_fraction | 1.0 | - | - | - |
| feature_fraction_bynode | 0.06363636363636363 | - | - | - |
| learning_rate | 1.0 | - | - | - |
| max_bin | 128 | 128 | 128 | - |
| max_depth | 8 | 8 | 8 | 8 |
| max_features | - | "sqrt" | "sqrt" | "sqrt" |
| max_leaves | 256 | -1 | -1 | null |
| max_samples | - | 1.0 | 1.0 | 1.0 |
| min_child_weight | 0.0 | - | - | - |
| min_samples_leaf | 1 | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| reg_alpha | 0.0 | - | - | - |
| reg_lambda | 0.0 | - | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | 0.632 | - | - | - |

accepted difference: ours-ab class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 119.8 | 119.8..119.8 | 1 | - | - | - | auc=0.945385, auc_matches_fit=True, logloss=0.182017, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 81.0 | 81.0..81.0 | 1 | - | - | - | auc=0.945385, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.182017, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | COMPARABLE | ok |
| sklearn-rf-cpu | test | 500000 | 990.6 | 990.6..990.6 | 1 | 0.121 | 0.082 | - | auc=0.945284, auc_matches_fit=True, logloss=0.182344, logloss_matches_fit=True | yes | COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 662.9 | 662.9..662.9 | 1 | 0.181 | 0.122 | - | auc=0.945361, auc_matches_fit=True, logloss=0.195422, logloss_matches_fit=True | yes | COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 208.2 | 208.2..208.2 | 1 | - | - | - | - | yes | COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 159.8 | 159.8..159.8 | 1 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | COMPARABLE | ok |
| sklearn-rf-cpu | large | 1000000 | 1906.1 | 1906.1..1906.1 | 1 | 0.109 | 0.084 | - | - | yes | COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1312.1 | 1312.1..1312.1 | 1 | 0.159 | 0.122 | - | - | yes | COMPARABLE | ok |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, sklearn-rf-cpu: sklearn RandomForestClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

### rf / taxi (rows full, shape taxi-4110786x16)

race: done, driver rc 0, log `raw/trees/rf.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10935.5 | 10935.5..10935.5 | 1 | - | - | - | 2480.0 | - | auc=0.617838, logloss=0.525953 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10796.4 | 10796.4..10796.4 | 1 | - | - | - | 2536.1 | - | auc=0.617838, logloss=0.525953 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 74151.1 | 74151.1..74151.1 | 1 | 0.147 | 0.146 | - | 6064.0 | - | auc=0.617678, logloss=0.525532 | yes | NOT-COMPARABLE | - | ok (measured this run) |
| lightgbm-cpu | lightgbm | cpu | opponent | 53864.4 | 53864.4..53864.4 | 1 | 0.203 | 0.200 | - | 1339.0 | - | auc=0.617040, logloss=0.526421 | yes | NOT-COMPARABLE | - | ok (measured this run) |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:101602,ours:123231,ours-ab:123231,sklearn-rf-cpu:89688 spread=0.2722 verdict=NOT-COMPARABLE`

config: NVIDIA gbm-bench, skrf/cumlrf: max_depth 8, n_estimators 500 (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | lightgbm-cpu | ours | ours-ab | sklearn-rf-cpu |
|---|---||---|---||---|---||---|---|
| library (source) | lightgbm (get_params) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| boosting_type | "rf" | - | - | - |
| bootstrap | - | true | true | true |
| class_weight | null | null | null | null |
| criterion | - | "gini" | "gini" | "gini" |
| feature_fraction | 1.0 | - | - | - |
| feature_fraction_bynode | 0.25 | - | - | - |
| learning_rate | 1.0 | - | - | - |
| max_bin | 128 | 128 | 128 | - |
| max_depth | 8 | 8 | 8 | 8 |
| max_features | - | "sqrt" | "sqrt" | "sqrt" |
| max_leaves | 256 | -1 | -1 | null |
| max_samples | - | 1.0 | 1.0 | 1.0 |
| min_child_weight | 0.0 | - | - | - |
| min_samples_leaf | 1 | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 500 | 500 | 500 | 500 |
| reg_alpha | 0.0 | - | - | - |
| reg_lambda | 0.0 | - | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | 0.632 | - | - | - |

accepted difference: ours-ab class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | test | 500000 | 60.8 | 60.8..60.8 | 1 | - | - | - | auc=0.617838, auc_matches_fit=True, logloss=0.525953, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | test | 500000 | 52.7 | 52.7..52.7 | 1 | - | - | - | auc=0.617838, auc_matches_fit=True, bits_equal_vs_ours_identical=True, logloss=0.525953, logloss_matches_fit=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-rf-cpu | test | 500000 | 390.4 | 390.4..390.4 | 1 | 0.156 | 0.135 | - | auc=0.617678, auc_matches_fit=True, logloss=0.525532, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | test | 500000 | 612.2 | 612.2..612.2 | 1 | 0.099 | 0.086 | - | auc=0.617040, auc_matches_fit=True, logloss=0.526421, logloss_matches_fit=True | yes | NOT-COMPARABLE | ok |
| mojolearn IDENTICAL | large | 1000000 | 123.2 | 123.2..123.2 | 1 | - | - | - | - | yes | NOT-COMPARABLE | ok |
| mojolearn FAST | large | 1000000 | 105.0 | 105.0..105.0 | 1 | - | - | - | bits_equal_vs_ours_identical=True, max_abs_diff_vs_ours_identical=0.000000 | yes | NOT-COMPARABLE | ok |
| sklearn-rf-cpu | large | 1000000 | 774.6 | 774.6..774.6 | 1 | 0.159 | 0.136 | - | - | yes | NOT-COMPARABLE | ok |
| lightgbm-cpu | large | 1000000 | 1223.9 | 1223.9..1223.9 | 1 | 0.101 | 0.086 | - | - | yes | NOT-COMPARABLE | ok |

inference call, ours: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, ours-ab: mojolearn RandomForestClassifier.predict_proba(X), column 1

inference call, sklearn-rf-cpu: sklearn RandomForestClassifier.predict_proba(X), n_jobs -1, column 1

inference call, lightgbm-cpu: lightgbm Booster.predict(X) (probability; LightGBM predicts on the CPU whatever device trained it)

## Classical

### dbscan / istella (rows full, shape 1000000x220)

race: done, driver rc 0, log `logs/classical.dbscan.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 255151.4 | 255151.4..255151.4 | 1 | - | - | - | 4154.2 | - | n_clusters=40131, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 247363.7 | 247363.7..247363.7 | 1 | - | - | - | 4119.5 | - | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 248662.7 | 248662.7..248662.7 | 1 | 1.026 | 0.995 | - | 5648.5 | - | ari_vs_ours=1.000000, n_clusters=40131, noise_agreement_vs_ours=1.000000, noise_fraction=0.219391, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T18:37:38Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows: dbscan block: 1,000,000 rows, standardized. Timed: fit.

mismatch: seed: no arm has a seed argument (deterministic)

mismatch: algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' (its default), scikit-learn 'brute' (the cuML benchmark's cpu_args; it has no 'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'

mismatch: leaf_size=30 and n_jobs=-1: scikit-learn only

config: cuML benchmark (RAPIDS), DBSCAN (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast |
|---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) |
| algorithm | "rbc" | "rbc" |
| eps | 3.0 | 3.0 |
| metric | "euclidean" | "euclidean" |
| min_samples | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" |

### dbscan / taxi (rows full, shape 1000000x11)

race: failed, driver rc 0, log `logs/classical.dbscan.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 758590.2 | 758590.2..758590.2 | 1 | - | - | - | 8888.6 | - | n_clusters=36, noise_fraction=0.000174, rows=1000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | HOST-MEMORY(killed at 230.2 GB: the driver's process tree held 247.5 GB, over 90% of the box's 274.9 GB) (measured this run) |

memory, ours, sklearn-cpu: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

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

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| sklearn-cpu | scikit-learn | cpu | opponent | 651128.7 | 651128.7..651128.7 | 1 | - | - | - | 1389.4 | - | n_clusters=52, noise_fraction=0.252570, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m cells and hierarchy's PAIRWISE connectivity refuses past that bound (their value_id) |
| mojolearn FAST | mojolearn | gpu | fast | 1009.6 | 1009.6..1009.6 | 1 | - | - | - | 516.0 | - | n_clusters=160, noise_fraction=0.142220, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 31945.7 | 31945.7..31945.7 | 1 | - | 0.032 | - | 260.5 | - | n_clusters=161, noise_fraction=0.134620, rows=100000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical.kde.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 179.5 | 179.5..179.5 | 1 | - | - | - | 1256.2 | - | mean_log_likelihood=-222.270586, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1567.5 | 1567.5..1567.5 | 1 | - | - | - | 487.8 | - | mean_log_likelihood=-222.270586, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 54439.8 | 54439.8..54439.8 | 1 | 0.003 | 0.029 | - | 402.3 | - | mean_log_likelihood=-226.977403, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical.kde.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 115.9 | 115.9..115.9 | 1 | - | - | - | 1088.2 | - | mean_log_likelihood=-14.826460, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 55.3 | 55.3..55.3 | 1 | - | - | - | 324.5 | - | mean_log_likelihood=-14.826460, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6994.8 | 6994.8..6994.8 | 1 | 0.017 | 0.008 | - | 148.9 | - | mean_log_likelihood=-14.826437, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2117.9 | 2117.9..2117.9 | 1 | - | - | - | 4315.8 | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1502.3 | 1502.3..1502.3 | 1 | - | - | - | 4317.7 | - | inertia=6.051e+17, inertia_over_ours=1.000000, n_iter=33 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3288.1 | 3288.1..3288.1 | 1 | 0.644 | 0.457 | - | 5862.2 | - | inertia=5.959e+17, inertia_over_ours=0.984774, n_iter=36 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 7789.9 | 7789.9..7789.9 | 1 | 0.272 | 0.193 | - | 9542.5 | 6912.7 | inertia=5.991e+17, inertia_over_ours=0.990156, n_iter=68 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | 500000 | 130.2 | 130.2..130.2 | 1 | - | - | - | eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 124.4 | 124.4..124.4 | 1 | - | - | - | agreement_vs_ours=1.000000, bits_equal_vs_ours=True, eval_inertia=1.417e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 57.5 | 57.5..57.5 | 1 | 2.263 | 2.162 | - | agreement_vs_ours=0.000000, bits_equal_vs_ours=False, eval_inertia=1.412e+17, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 45.4 | 45.4..45.4 | 1 | 2.871 | 2.742 | - | agreement_vs_ours=0.001558, bits_equal_vs_ours=False, eval_inertia=1.402e+17, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### kmeans / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 961.7 | 961.7..961.7 | 1 | - | - | - | 933.9 | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1030.5 | 1030.5..1030.5 | 1 | - | - | - | 932.9 | - | inertia=3.093e+08, inertia_over_ours=1.000000, n_iter=91 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1945.8 | 1945.8..1945.8 | 1 | 0.494 | 0.530 | - | 605.6 | - | inertia=3.166e+08, inertia_over_ours=1.023705, n_iter=100 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 2860.3 | 2860.3..2860.3 | 1 | 0.336 | 0.360 | - | 1932.8 | 1210.7 | inertia=3.06e+08, inertia_over_ours=0.989329, n_iter=71 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | 500000 | 39.3 | 39.3..39.3 | 1 | - | - | - | eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 36.7 | 36.7..36.7 | 1 | - | - | - | agreement_vs_ours=1.000000, bits_equal_vs_ours=True, eval_inertia=4.824e+07, label_agreement_own_centers=0.999998 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 15.6 | 15.6..15.6 | 1 | 2.516 | 2.348 | - | agreement_vs_ours=0.022278, bits_equal_vs_ours=False, eval_inertia=5.05e+07, label_agreement_own_centers=1.000000 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | 38.7 | 38.7..38.7 | 1 | 1.014 | 0.946 | - | agreement_vs_ours=0.040516, bits_equal_vs_ours=False, eval_inertia=4.572e+07, label_agreement_own_centers=1.000000 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | ok |

inference call, ours: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn KMeans.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn KMeans.predict(Xq), host rows in, host result out

inference call, torch-gpu: torch chunked addmm(//c//^2, Xq, c.T, alpha -2).argmin over the fitted centers; Xq uploaded before the clock, which ends at the device synchronize

### knn / istella (rows full, shape 400000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 772.2 | 772.2..772.2 | 1 | - | - | - | 1712.5 | - | recall_at_k=0.976250, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1605.7 | 1605.7..1605.7 | 1 | - | - | - | 1622.5 | - | recall_at_k=0.976613, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 533.5 | 533.5..533.5 | 1 | 1.447 | 3.010 | - | 566.3 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 674.5 | 674.5..674.5 | 1 | 1.145 | 2.380 | - | 9659.6 | 8892.5 | recall_at_k=0.979973, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 732.8 | 732.8..732.8 | 1 | - | - | - | 748.5 | - | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 225.8 | 225.8..225.8 | 1 | - | - | - | 948.9 | - | recall_at_k=0.999754, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 423.0 | 423.0..423.0 | 1 | 1.733 | 0.534 | - | 202.5 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-gpu | torch | gpu | opponent | 650.3 | 650.3..650.3 | 1 | 1.127 | 0.347 | - | 9331.1 | 8886.5 | recall_at_k=0.999738, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical.ols.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1151.4 | 1151.4..1151.4 | 1 | - | - | - | 9340.0 | - | finite=True, r2=0.331944, rmse=0.682027 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1293.2 | 1293.2..1293.2 | 1 | - | - | - | 9339.6 | - | finite=True, r2=0.321092, rmse=0.687544 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3214.4 | 3214.4..3214.4 | 1 | 0.358 | 0.402 | - | 5748.7 | - | finite=True, r2=0.001881, rmse=0.833655 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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
| mojolearn IDENTICAL | Xq | 500000 | 72.6 | 72.6..72.6 | 1 | - | - | - | predict_max_rel_err_own_fp64=9.581e-07, r2_eval=0.331944, rmse_eval=0.682027 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 84.3 | 84.3..84.3 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=2.693979, predict_max_rel_err_own_fp64=8.387e-07, r2_eval=0.321092, rmse_eval=0.687544 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 51.5 | 51.5..51.5 | 1 | 1.408 | 1.635 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=7.010309, predict_max_rel_err_own_fp64=1.18e-07, r2_eval=0.001881, rmse_eval=0.833655 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: predict(Xq)

### ols / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 112.5 | 112.5..112.5 | 1 | - | - | - | 1223.5 | - | finite=True, r2=0.908837, rmse=4.696466 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 68.9 | 68.9..68.9 | 1 | - | - | - | 1222.2 | - | finite=True, r2=0.908838, rmse=4.696444 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 248.0 | 248.0..248.0 | 1 | 0.454 | 0.278 | - | 602.5 | - | finite=True, r2=0.724848, rmse=8.159214 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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
| mojolearn IDENTICAL | Xq | 500000 | 32.4 | 32.4..32.4 | 1 | - | - | - | predict_max_rel_err_own_fp64=8.387e-08, r2_eval=0.908837, rmse_eval=4.696466 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 32.4 | 32.4..32.4 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.026733, predict_max_rel_err_own_fp64=1.15e-07, r2_eval=0.908838, rmse_eval=4.696444 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 6.9 | 6.9..6.9 | 1 | 4.715 | 4.712 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=197.959106, predict_max_rel_err_own_fp64=1.016e-07, r2_eval=0.724848, rmse_eval=8.159213 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn LinearRegression.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn LinearRegression.predict(Xq), host rows in, host result out

inference call, torch-gpu: predict(Xq)

### pca / istella (rows full, shape 2043304x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 719.3 | 719.3..719.3 | 1 | - | - | - | 7638.8 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 924.2 | 924.2..924.2 | 1 | - | - | - | 7639.5 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 229.0 | 229.0..229.0 | 1 | 3.140 | 4.035 | - | 2285.9 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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
| mojolearn IDENTICAL | Xq | 500000 | 120.9 | 120.9..120.9 | 1 | - | - | - | transform_max_rel_err_own_fp64=3.75e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 119.7 | 119.7..119.7 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=62156.501953, transform_max_rel_err_own_fp64=3.781e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 59.5 | 59.5..59.5 | 1 | 2.034 | 2.014 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=87980.788086, transform_max_rel_err_own_fp64=3.839e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, ours-fast: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: transform(Xq)

### pca / taxi (rows full, shape 4000000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 105.2 | 105.2..105.2 | 1 | - | - | - | 1054.8 | - | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 50.8 | 50.8..50.8 | 1 | - | - | - | 1056.8 | - | explained_variance_ratio_sum=0.999997 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 126.5 | 126.5..126.5 | 1 | 0.832 | 0.402 | - | 341.6 | - | explained_variance_ratio_sum=0.999996 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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
| mojolearn IDENTICAL | Xq | 500000 | 45.4 | 45.4..45.4 | 1 | - | - | - | transform_max_rel_err_own_fp64=1.085e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 500000 | 43.6 | 43.6..43.6 | 1 | - | - | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.003017, transform_max_rel_err_own_fp64=1.186e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 500000 | 22.8 | 22.8..22.8 | 1 | 1.994 | 1.916 | - | bits_equal_vs_ours=False, max_abs_diff_vs_ours=0.719910, transform_max_rel_err_own_fp64=1.161e-07 | yes | LIKE-FOR-LIKE-SPAN | ok |
| torch-gpu | Xq | 500000 | - | - | 0 | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | REFUSED(no_fit: null) |

inference call, ours: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, ours-fast: mojolearn PCA.transform(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn PCA.transform(Xq), host rows in, host result out

inference call, torch-gpu: transform(Xq)

### svc / istella (rows full, shape 10000x220)

race: done, driver rc 0, log `logs/classical.svc.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 108.6 | 108.6..108.6 | 1 | - | - | - | 425.2 | - | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 173.0 | 173.0..173.0 | 1 | - | - | - | 466.1 | - | accuracy=0.922200, n_support=2401 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1059.8 | 1059.8..1059.8 | 1 | 0.102 | 0.163 | - | 306.1 | - | accuracy=0.922200, n_support=2400 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | 10000 | 46.4 | 46.4..46.4 | 1 | - | - | - | accuracy_eval=0.922200 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 10000 | 75.3 | 75.3..75.3 | 1 | - | - | - | accuracy_eval=0.922200, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 10000 | 1831.7 | 1831.7..1831.7 | 1 | 0.025 | 0.041 | - | accuracy_eval=0.922200, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn SVC.predict(Xq), host rows in, host result out

### svc / taxi (rows full, shape 10000x11)

race: done, driver rc 0, log `logs/classical.svc.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 758.6 | 758.6..758.6 | 1 | - | - | - | 571.3 | - | accuracy=0.767500, n_support=5527 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 509.4 | 509.4..509.4 | 1 | - | - | - | 415.1 | - | accuracy=0.767500, n_support=5525 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2472.5 | 2472.5..2472.5 | 1 | 0.307 | 0.206 | - | 379.7 | - | accuracy=0.767500, n_support=5672 | yes | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | 10000 | 65.4 | 65.4..65.4 | 1 | - | - | - | accuracy_eval=0.767500 | yes | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | 10000 | 25.5 | 25.5..25.5 | 1 | - | - | - | accuracy_eval=0.767500, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | 10000 | 1348.6 | 1348.6..1348.6 | 1 | 0.048 | 0.019 | - | accuracy_eval=0.767500, agreement_vs_ours=1.000000, bits_equal_vs_ours=True | yes | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, ours-fast: mojolearn SVC.predict(Xq), host rows in, host result out

inference call, sklearn-cpu: sklearn SVC.predict(Xq), host rows in, host result out

## Classical, wave 2

### agglomerative / istella (rows full, shape X 10000x220)

race: done, driver rc 0, log `logs/classical2.agglomerative.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 153.9 | 153.9..153.9 | 1 | - | - | - | 1677.5 | - | n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 140.6 | 140.6..140.6 | 1 | - | - | - | 1674.6 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4832.5 | 4832.5..4832.5 | 1 | 0.032 | 0.029 | - | 1089.0 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.716728 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.agglomerative.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 71.8 | 71.8..71.8 | 1 | - | - | - | 782.3 | - | n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 68.6 | 68.6..68.6 | 1 | - | - | - | 367.2 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 224.6 | 224.6..224.6 | 1 | 0.319 | 0.305 | - | 187.5 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.685524 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.arima.synthetic.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 99.3 | 99.3..99.3 | 1 | - | - | - | 575.6 | - | forecast_rmse=1.515518, insample_rmse=0.999342, mean_aic=5680.976967, mean_llf=-2836.488483 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 102.3 | 102.3..102.3 | 1 | - | - | - | 571.8 | - | forecast_rmse=1.515540, insample_rmse=0.999341, mean_aic=5680.971687, mean_llf=-2836.485844 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 303.1 | 303.1..303.1 | 1 | 0.328 | 0.338 | - | 142.3 | - | forecast_rmse=1.515423, insample_rmse=0.999338, mean_aic=5680.957160, mean_llf=-2836.478580 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.elasticnet.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2915.3 | 2915.3..2915.3 | 1 | - | - | - | 3061.8 | - | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 357.3 | 357.3..357.3 | 1 | - | - | - | 3866.0 | - | finite=True, r2=0.261554, rmse=0.717827 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1793.8 | 1793.8..1793.8 | 1 | 1.625 | 0.199 | - | 2748.0 | - | finite=True, r2=0.260922, rmse=0.718134 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.elasticnet.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 64.6 | 64.6..64.6 | 1 | - | - | - | 479.6 | - | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 59.8 | 59.8..59.8 | 1 | - | - | - | 513.0 | - | finite=True, r2=0.907378, rmse=4.847224 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 37.7 | 37.7..37.7 | 1 | 1.712 | 1.585 | - | 275.9 | - | finite=True, r2=0.907378, rmse=4.847223 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.ets.synthetic.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 205.0 | 205.0..205.0 | 1 | - | - | - | 339.9 | - | forecast_rmse=0.984392, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 189.4 | 189.4..189.4 | 1 | - | - | - | 338.8 | - | forecast_rmse=0.984473, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 884.4 | 884.4..884.4 | 1 | 0.232 | 0.214 | - | 140.6 | - | forecast_rmse=0.984418, insample_rmse=0.991812 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.gmm.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9176.3 | 9176.3..9176.3 | 1 | - | - | - | 2659.0 | - | bic=-3.851e+07, mean_log_likelihood=200.794403, n_iter=24 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6614.3 | 6614.3..6614.3 | 1 | - | - | - | 2113.0 | - | bic=-3.851e+07, mean_log_likelihood=200.794469, n_iter=24 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7479.2 | 7479.2..7479.2 | 1 | 1.227 | 0.884 | - | 1244.8 | - | bic=-3.901e+07, mean_log_likelihood=200.776340, n_iter=30 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 632.4 | 632.4..632.4 | 1 | - | - | - | 487.2 | - | bic=-3.67e+06, mean_log_likelihood=12.861940, n_iter=32 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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

race: done, driver rc 0, log `logs/classical2.gpc.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 801.4 | 801.4..801.4 | 1 | - | - | - | 1772.6 | - | accuracy=0.901333, logloss=0.232590, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 509.3 | 509.3..509.3 | 1 | - | - | - | 1748.2 | - | accuracy=0.901333, logloss=0.232592, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1162.4 | 1162.4..1162.4 | 1 | 0.689 | 0.438 | - | 1269.3 | - | accuracy=0.901333, logloss=0.232597, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.gpc.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 576.9 | 576.9..576.9 | 1 | - | - | - | 870.9 | - | accuracy=0.761000, logloss=0.541286, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 407.8 | 407.8..407.8 | 1 | - | - | - | 846.8 | - | accuracy=0.761000, logloss=0.541355, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 640.9 | 640.9..640.9 | 1 | 0.900 | 0.636 | - | 381.9 | - | accuracy=0.761000, logloss=0.541358, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.gpr.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 258.9 | 258.9..258.9 | 1 | - | - | - | 1705.6 | - | finite=True, mean_log_predictive_density=-9.285754, r2=0.235346, rmse=0.760439 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 214.7 | 214.7..214.7 | 1 | - | - | - | 1735.9 | - | finite=True, mean_log_predictive_density=-9.285319, r2=0.235374, rmse=0.760426 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 495.1 | 495.1..495.1 | 1 | 0.523 | 0.434 | - | 1255.3 | - | finite=True, mean_log_predictive_density=-9.287148, r2=0.235368, rmse=0.760428 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.gpr.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 232.1 | 232.1..232.1 | 1 | - | - | - | 852.7 | - | finite=True, mean_log_predictive_density=-311.458394, r2=0.889630, rmse=5.041639 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 199.4 | 199.4..199.4 | 1 | - | - | - | 814.1 | - | finite=True, mean_log_predictive_density=-311.454431, r2=0.889630, rmse=5.041637 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 136.8 | 136.8..136.8 | 1 | 1.696 | 1.457 | - | 367.9 | - | finite=True, mean_log_predictive_density=-311.539594, r2=0.889629, rmse=5.041653 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.ivf.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4865.8 | 4865.8..4865.8 | 1 | - | - | - | 2044.4 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2528.4 | 2528.4..2528.4 | 1 | - | - | - | 2499.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 4983.1 | 4983.1..4983.1 | 1 | 0.976 | 0.507 | - | 1526.4 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.ivf.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 554.7 | 554.7..554.7 | 1 | - | - | - | 708.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 480.1 | 480.1..480.1 | 1 | - | - | - | 726.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 3754.8 | 3754.8..3754.8 | 1 | 0.148 | 0.128 | - | 157.9 | - | - | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.kernel-ridge.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 647.3 | 647.3..647.3 | 1 | - | - | - | 2118.7 | - | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 298.9 | 298.9..298.9 | 1 | - | - | - | 2082.0 | - | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 778.1 | 778.1..778.1 | 1 | 0.832 | 0.384 | - | 2055.9 | - | finite=True, r2=0.385427, rmse=0.646407 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.kernel-ridge.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 673.7 | 673.7..673.7 | 1 | - | - | - | 1350.5 | - | finite=True, r2=0.726543, rmse=8.330373 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 220.0 | 220.0..220.0 | 1 | - | - | - | 1167.3 | - | finite=True, r2=0.726543, rmse=8.330374 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 757.5 | 757.5..757.5 | 1 | 0.889 | 0.290 | - | 1170.2 | - | finite=True, r2=0.726542, rmse=8.330380 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.knn-clf.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 214.4 | 214.4..214.4 | 1 | - | - | - | 1955.1 | - | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 827.5 | 827.5..827.5 | 1 | - | - | - | 1889.8 | - | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 308.8 | 308.8..308.8 | 1 | 0.694 | 2.680 | - | 1312.6 | - | accuracy=0.926250 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.knn-clf.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 57.1 | 57.1..57.1 | 1 | - | - | - | 602.7 | - | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 45.5 | 45.5..45.5 | 1 | - | - | - | 529.0 | - | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 125.1 | 125.1..125.1 | 1 | 0.456 | 0.364 | - | 224.8 | - | accuracy=0.741750 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.knn-reg.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 200.2 | 200.2..200.2 | 1 | - | - | - | 1947.7 | - | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 820.2 | 820.2..820.2 | 1 | - | - | - | 1880.0 | - | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 261.3 | 261.3..261.3 | 1 | 0.766 | 3.139 | - | 1313.6 | - | finite=True, r2=0.418145, rmse=0.625388 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.knn-reg.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 31.6 | 31.6..31.6 | 1 | - | - | - | 596.5 | - | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 20.7 | 20.7..20.7 | 1 | - | - | - | 513.6 | - | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 130.4 | 130.4..130.4 | 1 | 0.242 | 0.158 | - | 219.8 | - | finite=True, r2=0.937323, rmse=3.842028 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.lasso.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8482.0 | 8482.0..8482.0 | 1 | - | - | - | 3062.4 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 364.6 | 364.6..364.6 | 1 | - | - | - | 3864.8 | - | finite=True, r2=0.310472, rmse=0.693643 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3842.1 | 3842.1..3842.1 | 1 | 2.208 | 0.095 | - | 2745.8 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.lasso.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 70.7 | 70.7..70.7 | 1 | - | - | - | 483.0 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 60.3 | 60.3..60.3 | 1 | - | - | - | 513.6 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 44.5 | 44.5..44.5 | 1 | 1.587 | 1.355 | - | 278.1 | - | finite=True, r2=0.908995, rmse=4.804745 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.linearsvc.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 762.6 | 762.6..762.6 | 1 | - | - | - | 2163.6 | - | accuracy=0.923480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 754.3 | 754.3..754.3 | 1 | - | - | - | 2159.3 | - | accuracy=0.923230 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 594659.6 | 594659.6..594659.6 | 1 | 0.001 | 0.001 | - | 6163.2 | - | accuracy=0.923540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.linearsvc.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 108.7 | 108.7..108.7 | 1 | - | - | - | 475.9 | - | accuracy=0.763330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 107.1 | 107.1..107.1 | 1 | - | - | - | 462.4 | - | accuracy=0.763330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 429.0 | 429.0..429.0 | 1 | 0.253 | 0.250 | - | 469.3 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.linearsvr.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 952.3 | 952.3..952.3 | 1 | - | - | - | 2098.1 | - | finite=True, r2=-0.106754, rmse=0.878792 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 905.9 | 905.9..905.9 | 1 | - | - | - | 2099.9 | - | finite=True, r2=-0.106754, rmse=0.878792 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 691569.7 | 691569.7..691569.7 | 1 | 0.001 | 0.001 | - | 6137.4 | - | finite=True, r2=-0.025729, rmse=0.846012 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.linearsvr.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 101.5 | 101.5..101.5 | 1 | - | - | - | 422.2 | - | finite=True, r2=0.899813, rmse=5.041302 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 191.9 | 191.9..191.9 | 1 | - | - | - | 421.8 | - | finite=True, r2=0.899814, rmse=5.041262 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 83289.0 | 83289.0..83289.0 | 1 | 0.001 | 0.002 | - | 511.0 | - | finite=True, r2=0.899803, rmse=5.041552 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.logreg.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3796.3 | 3796.3..3796.3 | 1 | - | - | - | 2108.3 | - | accuracy=0.924540, logloss=0.181245, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3922.5 | 3922.5..3922.5 | 1 | - | - | - | 2109.8 | - | accuracy=0.924560, logloss=0.181242, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28415.0 | 28415.0..28415.0 | 1 | 0.134 | 0.138 | - | 2939.8 | - | accuracy=0.924470, logloss=0.181337, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.logreg.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 133.0 | 133.0..133.0 | 1 | - | - | - | 427.5 | - | accuracy=0.763340, logloss=0.538984, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 120.1 | 120.1..120.1 | 1 | - | - | - | 425.2 | - | accuracy=0.763310, logloss=0.538987, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 304.0 | 304.0..304.0 | 1 | 0.437 | 0.395 | - | 304.3 | - | accuracy=0.763320, logloss=0.538980, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.nystroem.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 467.5 | 467.5..467.5 | 1 | - | - | - | 1472.1 | - | kernel_rel_error=0.038958 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.nystroem.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('km_basis_indices: n_samples 100000 exceeds KM_MAX_BASIS_POOL = 4096. The rank pass counts a total order over every pair of rows, which is n_samples^2 host comparisons. To close t) |
| sklearn-cpu | scikit-learn | cpu | opponent | 295.7 | 295.7..295.7 | 1 | - | - | - | 492.4 | - | kernel_rel_error=0.044370 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.rbf-sampler.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 53.5 | 53.5..53.5 | 1 | - | - | - | 1903.0 | - | kernel_rel_error=0.141980 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 76.8 | 76.8..76.8 | 1 | - | - | - | 1853.2 | - | kernel_rel_error=0.141980 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 47.5 | 47.5..47.5 | 1 | 1.125 | 1.617 | - | 1346.5 | - | kernel_rel_error=0.137405 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.rbf-sampler.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 26.4 | 26.4..26.4 | 1 | - | - | - | 926.7 | - | kernel_rel_error=0.108549 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 27.3 | 27.3..27.3 | 1 | - | - | - | 734.0 | - | kernel_rel_error=0.108549 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 41.0 | 41.0..41.0 | 1 | 0.645 | 0.667 | - | 386.5 | - | kernel_rel_error=0.083775 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.ridge.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 822.1 | 822.1..822.1 | 1 | - | - | - | 5450.1 | - | finite=True, r2=0.328682, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 844.7 | 844.7..844.7 | 1 | - | - | - | 5452.2 | - | finite=True, r2=0.320451, rmse=0.688606 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6052.6 | 6052.6..6052.6 | 1 | 0.136 | 0.140 | - | 8646.4 | - | finite=True, r2=0.328676, rmse=0.684426 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.ridge.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 23.1 | 23.1..23.1 | 1 | - | - | - | 581.6 | - | finite=True, r2=0.908983, rmse=4.805042 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 28.1 | 28.1..28.1 | 1 | - | - | - | 582.8 | - | finite=True, r2=0.908983, rmse=4.805048 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34.8 | 34.8..34.8 | 1 | 0.664 | 0.809 | - | 230.4 | - | finite=True, r2=0.908988, rmse=4.804916 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.spectral-embedding.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 460.5 | 460.5..460.5 | 1 | - | - | - | 484.7 | - | trustworthiness_k15=0.799378 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 435.2 | 435.2..435.2 | 1 | - | - | - | 445.3 | - | trustworthiness_k15=0.823640 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8207.4 | 8207.4..8207.4 | 1 | 0.056 | 0.053 | - | 413.4 | - | trustworthiness_k15=0.812688 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.spectral-embedding.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 239.1 | 239.1..239.1 | 1 | - | - | - | 406.9 | - | trustworthiness_k15=0.884889 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 110.2 | 110.2..110.2 | 1 | - | - | - | 393.2 | - | trustworthiness_k15=0.895236 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2337.6 | 2337.6..2337.6 | 1 | 0.102 | 0.047 | - | 200.4 | - | trustworthiness_k15=0.898011 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.spectral.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 396.8 | 396.8..396.8 | 1 | - | - | - | 1347.9 | - | n_clusters=8, silhouette=0.147668 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 249.0 | 249.0..249.0 | 1 | - | - | - | 1336.1 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.147668 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1574.5 | 1574.5..1574.5 | 1 | 0.252 | 0.158 | - | 1199.7 | - | ari_vs_ours=0.999826, n_clusters=8, silhouette=0.147699 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.spectral.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 445.5 | 445.5..445.5 | 1 | - | - | - | 448.4 | - | n_clusters=8, silhouette=0.039910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 158.3 | 158.3..158.3 | 1 | - | - | - | 447.8 | - | ari_vs_ours=0.997088, n_clusters=8, silhouette=0.039888 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 933.7 | 933.7..933.7 | 1 | 0.477 | 0.170 | - | 233.1 | - | ari_vs_ours=0.582313, n_clusters=8, silhouette=0.089894 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.svr.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 120.3 | 120.3..120.3 | 1 | - | - | - | 1355.9 | - | finite=True, r2=0.318258, rmse=0.680816 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 182.7 | 182.7..182.7 | 1 | - | - | - | 1398.5 | - | finite=True, r2=0.318258, rmse=0.680816 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1425.6 | 1425.6..1425.6 | 1 | 0.084 | 0.128 | - | 1192.8 | - | finite=True, r2=0.318248, rmse=0.680821 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.svr.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 112.6 | 112.6..112.6 | 1 | - | - | - | 616.4 | - | finite=True, r2=0.767551, rmse=7.680395 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 92.2 | 92.2..92.2 | 1 | - | - | - | 461.3 | - | finite=True, r2=0.767550, rmse=7.680409 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1142.1 | 1142.1..1142.1 | 1 | 0.099 | 0.081 | - | 520.9 | - | finite=True, r2=0.767550, rmse=7.680414 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.tsvd.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 391.1 | 391.1..391.1 | 1 | - | - | - | 3707.7 | - | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.002554 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 345.4 | 345.4..345.4 | 1 | - | - | - | 3707.5 | - | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.002554 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1388.5 | 1388.5..1388.5 | 1 | 0.282 | 0.249 | - | 2020.8 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.000122 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.tsvd.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 35.6 | 35.6..35.6 | 1 | - | - | - | 595.7 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 30.2 | 30.2..30.2 | 1 | - | - | - | 592.9 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 150.6 | 150.6..150.6 | 1 | 0.237 | 0.201 | - | 384.9 | - | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.003257 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.umap.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1487.3 | 1487.3..1487.3 | 1 | - | - | - | 510.9 | - | trustworthiness_k15=0.979906 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 937.2 | 937.2..937.2 | 1 | - | - | - | 479.0 | - | trustworthiness_k15=0.982091 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 9061.0 | 9061.0..9061.0 | 1 | 0.164 | 0.103 | - | 580.4 | - | trustworthiness_k15=0.978822 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1438.9 | 1438.9..1438.9 | 1 | 1.034 | 0.651 | - | 609.2 | - | trustworthiness_k15=0.977078 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/classical2.umap.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 582.9 | 582.9..582.9 | 1 | - | - | - | 418.9 | - | trustworthiness_k15=0.990480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 345.0 | 345.0..345.0 | 1 | - | - | - | 388.7 | - | trustworthiness_k15=0.991488 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 8010.7 | 8010.7..8010.7 | 1 | 0.073 | 0.043 | - | 497.5 | - | trustworthiness_k15=0.989525 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1348.4 | 1348.4..1348.4 | 1 | 0.432 | 0.256 | - | 523.3 | - | trustworthiness_k15=0.991876 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/neural.gemm-bf16.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 67.7 | 67.7..67.7 | 1 | - | - | - | 933.1 | - | max_rel_err_vs_fp64=1.155e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-bf16 | torch | gpu | opponent | 9.3 | 9.3..9.3 | 1 | 7.317 | - | - | 1706.3 | 1032.7 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 10.2 | 10.2..10.2 | 1 | 6.619 | - | - | 1868.0 | 1032.7 | max_abs_diff_vs_ours=0.999023, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-eager-bf16 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### gemm-int8 / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm-int8.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1954.7 | 1954.7..1954.7 | 1 | - | - | - | 620.0 | - | max_rel_err_vs_fp64=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours |
|---|---|
| library (source) | mojolearn (declared) |
| seed | "none (deterministic)" |

### gemm / gaussian (neural shape full: 4096x4096x4096)

race: done, driver rc 0, log `logs/neural.gemm.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 72.1 | 72.1..72.1 | 1 | - | - | - | 805.3 | - | max_rel_err_vs_fp64=2.399e-07 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 11.5 | 11.5..11.5 | 1 | 6.270 | - | - | 1629.3 | 1024.5 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.843e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 11.8 | 11.8..11.8 | 1 | 6.135 | - | - | 1795.0 | 1024.5 | max_abs_diff_vs_ours=0.001038, max_rel_diff_vs_ours=2.866e-06, max_rel_err_vs_fp64=2.843e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 10.8 | 10.8..10.8 | 1 | 6.681 | - | - | 1643.8 | 1032.7 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 11.4 | 11.4..11.4 | 1 | 6.321 | - | - | 1809.0 | 1032.7 | max_abs_diff_vs_ours=1.358337, max_rel_diff_vs_ours=0.003751, max_rel_err_vs_fp64=0.003752 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-forward / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-forward.bytes.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 250.1 | 250.1..250.1 | 1 | - | - | - | 4596.7 | - | mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 25.5 | 25.5..25.5 | 1 | 9.812 | - | - | 1635.7 | 1048.5 | max_abs_diff_vs_ours=1.252e-06, max_rel_diff_vs_ours=1.718e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 20.8 | 20.8..20.8 | 1 | 12.031 | - | - | 1823.5 | 1048.7 | max_abs_diff_vs_ours=1.296e-06, max_rel_diff_vs_ours=1.78e-06, mean_nll=9.018733 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 26.3 | 26.3..26.3 | 1 | 9.496 | - | - | 1668.8 | 1080.7 | max_abs_diff_vs_ours=0.006710, max_rel_diff_vs_ours=0.009211, mean_nll=9.018650 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 12.5 | 12.5..12.5 | 1 | 19.957 | - | - | 1833.6 | 1048.7 | max_abs_diff_vs_ours=0.006710, max_rel_diff_vs_ours=0.009211, mean_nll=9.018662 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### lm-host-train-step / bytes (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-host-train-step.bytes.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 1659.1 | 1659.1..1659.1 | 1 | - | - | - | 2383.1 | - | loss_first_step=9.017858, loss_last_step=8.367768, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 131.9 | 131.9..131.9 | 1 | 12.583 | - | - | 843.5 | - | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.017857, loss_last_abs_diff_vs_ours=3.815e-06, loss_last_step=8.367764, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 4583.7 | 4583.7..4583.7 | 1 | 0.362 | - | - | 822.6 | - | loss_first_abs_diff_vs_ours=0.0001326, loss_first_step=9.017725, loss_last_abs_diff_vs_ours=0.0002737, loss_last_step=8.367495, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 4632.9 | 4632.9..4632.9 | 1 | 0.358 | - | - | 675.7 | - | loss_first_abs_diff_vs_ours=9.155e-05, loss_first_step=9.017766, loss_last_abs_diff_vs_ours=0.001106, loss_last_step=8.368875, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:14:09Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 175.8 | 175.8..175.8 | 1 | 9.437 | - | - | 704.0 | - | loss_first_abs_diff_vs_ours=9.537e-07, loss_first_step=9.017857, loss_last_abs_diff_vs_ours=1.907e-06, loss_last_step=8.367766, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:14:09Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, torch-cpu-compile-fp32, torch-cpu-compile-bf16, torch-cpu-eager-bf16, torch-cpu-eager-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| seed | "none (deterministic)" | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 |

### lm-infer / bytes (neural shape full: B1 L512 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-infer.bytes.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 1367.9 | 1367.9..1367.9 | 1 | - | - | - | 515.0 | - | mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 28.6 | 28.6..28.6 | 1 | 47.813 | - | - | 548.3 | - | max_abs_diff_vs_ours=1.013e-06, max_rel_diff_vs_ours=1.391e-06, mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 100.9 | 100.9..100.9 | 1 | 13.559 | - | - | 537.4 | - | max_abs_diff_vs_ours=0.006585, max_rel_diff_vs_ours=0.009040, mean_nll=9.017744 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 121.7 | 121.7..121.7 | 1 | 11.244 | - | - | 389.7 | - | max_abs_diff_vs_ours=0.006131, max_rel_diff_vs_ours=0.008417, mean_nll=9.017766 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:13:44Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 46.7 | 46.7..46.7 | 1 | 29.263 | - | - | 345.3 | - | max_abs_diff_vs_ours=9.239e-07, max_rel_diff_vs_ours=1.268e-06, mean_nll=9.017857 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:13:44Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, torch-cpu-compile-fp32, torch-cpu-compile-bf16, torch-cpu-eager-bf16, torch-cpu-eager-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### lm-train-step / bytes (neural shape full: B1 L2048 DM384 H6 KV6 HD64 FF1024 layers8 V8192)

race: done, driver rc 0, log `logs/neural.lm-train-step.bytes.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 220.4 | 220.4..220.4 | 1 | - | - | - | 4057.8 | - | loss_first_step=9.018733, loss_last_step=8.422411, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 81.9 | 81.9..81.9 | 1 | 2.691 | - | - | 3900.0 | 3226.7 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=9.018733, loss_last_abs_diff_vs_ours=9.537e-07, loss_last_step=8.422412, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 60.6 | 60.6..60.6 | 1 | 3.639 | - | - | 3010.0 | 2168.7 | loss_first_abs_diff_vs_ours=0.000000, loss_first_step=9.018733, loss_last_abs_diff_vs_ours=1.907e-06, loss_last_step=8.422413, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 85.9 | 85.9..85.9 | 1 | 2.566 | - | - | 2908.1 | 2226.7 | loss_first_abs_diff_vs_ours=0.003240, loss_first_step=9.021973, loss_last_abs_diff_vs_ours=0.004076, loss_last_step=8.418335, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 43.5 | 43.5..43.5 | 1 | 5.072 | - | - | 3025.8 | 2200.7 | loss_first_abs_diff_vs_ours=0.0002089, loss_first_step=9.018524, loss_last_abs_diff_vs_ours=0.002214, loss_last_step=8.420197, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/neural.mamba1-forward.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24.9 | 24.9..24.9 | 1 | - | - | - | 592.0 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 94.9 | 94.9..94.9 | 1 | 0.262 | - | - | 1740.7 | 1232.5 | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.189e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 99.8 | 99.8..99.8 | 1 | 0.249 | - | - | 1710.6 | 1200.7 | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.568e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba1-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba1-infer.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 78.7 | 78.7..78.7 | 1 | - | - | - | 79.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 97.6 | 97.6..97.6 | 1 | 0.807 | - | - | 326.7 | - | max_abs_diff_vs_ours=1.192e-07, max_rel_diff_vs_ours=5.948e-08 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 240.3 | 240.3..240.3 | 1 | 0.328 | - | - | 333.1 | - | max_abs_diff_vs_ours=5.15e-05, max_rel_diff_vs_ours=2.57e-05 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-eager-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba2-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-forward.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 32.7 | 32.7..32.7 | 1 | - | - | - | 483.8 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 227.0 | 227.0..227.0 | 1 | 0.144 | - | - | 7749.2 | 7248.8 | max_abs_diff_vs_ours=2.027e-06, max_rel_diff_vs_ours=6.89e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 225.9 | 225.9..225.9 | 1 | 0.145 | - | - | 7862.1 | 7248.8 | max_abs_diff_vs_ours=2.027e-06, max_rel_diff_vs_ours=6.89e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 190.3 | 190.3..190.3 | 1 | 0.172 | - | - | 6245.3 | 5744.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 176.2 | 176.2..176.2 | 1 | 0.185 | - | - | 6358.1 | 5744.8 | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, torch-eager-fp32, torch-compile-fp32, torch-eager-bf16, torch-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mamba2-infer / gaussian (neural shape full: B1 L512 DM384)

race: done, driver rc 0, log `logs/neural.mamba2-infer.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 52.8 | 52.8..52.8 | 1 | - | - | - | 100.5 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 163.7 | 163.7..163.7 | 1 | 0.322 | - | - | 1356.8 | - | max_abs_diff_vs_ours=1.192e-06, max_rel_diff_vs_ours=4.233e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 161.9 | 161.9..161.9 | 1 | 0.326 | - | - | 583.4 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 163.3 | 163.3..163.3 | 1 | 0.323 | - | - | 463.4 | - | max_abs_diff_vs_ours=0.005920, max_rel_diff_vs_ours=0.002102 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:00:56Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 165.8 | 165.8..165.8 | 1 | 0.318 | - | - | 1227.2 | - | max_abs_diff_vs_ours=1.192e-06, max_rel_diff_vs_ours=4.233e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:00:56Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, torch-cpu-compile-fp32, torch-cpu-compile-bf16, torch-cpu-eager-bf16, torch-cpu-eager-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mamba3-forward / gaussian (neural shape full: B1 L2048 DM384)

race: done, driver rc 0, log `logs/neural.mamba3-forward.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 30.5 | 30.5..30.5 | 1 | - | - | - | 473.7 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 76.7 | 76.7..76.7 | 1 | 0.397 | - | - | 1636.8 | 1096.8 | max_abs_diff_vs_ours=8.345e-07, max_rel_diff_vs_ours=3.651e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/reduction_utils.h>\\n#include <c10/metal/utils) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 80.7 | 80.7..80.7 | 1 | 0.377 | - | - | 1641.9 | 1096.8 | max_abs_diff_vs_ours=0.002095, max_rel_diff_vs_ours=0.0009164 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/neural.mamba3-infer.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 56.6 | 56.6..56.6 | 1 | - | - | - | 106.2 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 22.2 | 22.2..22.2 | 1 | 2.553 | - | - | 453.4 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.119e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 45.4 | 45.4..45.4 | 1 | 1.246 | - | - | 459.5 | - | max_abs_diff_vs_ours=0.002040, max_rel_diff_vs_ours=0.0009065 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 56.9 | 56.9..56.9 | 1 | 0.995 | - | - | 291.3 | - | max_abs_diff_vs_ours=0.002148, max_rel_diff_vs_ours=0.0009547 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:05:53Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 29.8 | 29.8..29.8 | 1 | 1.896 | - | - | 293.6 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=2.119e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:05:53Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, torch-cpu-compile-fp32, torch-cpu-compile-bf16, torch-cpu-eager-bf16, torch-cpu-eager-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 |

### mlp-infer / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-infer.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 0.2 | 0.2..0.2 | 1 | - | - | - | 47.3 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 0.5 | 0.5..0.5 | 1 | 0.464 | - | - | 150.4 | - | max_abs_diff_vs_ours=0.000000, max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 1.0 | 1.0..1.0 | 1 | 0.232 | - | - | 326.1 | - | max_abs_diff_vs_ours=0.000000, max_rel_diff_vs_ours=0.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 0.8 | 0.8..0.8 | 1 | 0.300 | - | - | 152.8 | - | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 1.1 | 1.1..1.1 | 1 | 0.219 | - | - | 325.3 | - | max_abs_diff_vs_ours=0.004285, max_rel_diff_vs_ours=0.003956 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, torch-cpu-eager-fp32, torch-cpu-compile-fp32, torch-cpu-eager-bf16, torch-cpu-compile-bf16: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | "none (deterministic)" | 7 | 7 | 7 | 7 |

### mlp-train-step / gaussian (neural shape full: rows256 8-16-3)

race: done, driver rc 0, log `logs/neural.mlp-train-step.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 17.9 | 17.9..17.9 | 1 | - | - | - | 314.3 | - | loss_first_step=1.160401, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 7.7 | 7.7..7.7 | 1 | 2.333 | - | - | 520.5 | 18.7 | loss_first_abs_diff_vs_ours=3.576e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 7.4 | 7.4..7.4 | 1 | 2.397 | - | - | 619.2 | 16.5 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.160401, loss_last_abs_diff_vs_ours=2.384e-07, loss_last_step=1.123361, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 4.2 | 4.2..4.2 | 1 | 4.283 | - | - | 523.7 | 18.7 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.3 | 4.3..4.3 | 1 | 4.185 | - | - | 619.6 | 16.7 | loss_first_abs_diff_vs_ours=9.656e-05, loss_first_step=1.160498, loss_last_abs_diff_vs_ours=0.0001006, loss_last_step=1.123461, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/neural.samba-forward.bytes.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 61.9 | 61.9..61.9 | 1 | - | - | - | 613.2 | - | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 42.8 | 42.8..42.8 | 1 | 1.447 | - | - | 727.2 | 168.8 | max_abs_diff_vs_ours=2.503e-06, max_rel_diff_vs_ours=1.407e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/error.h>\\n#include <c10/metal/reduction_utils) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 42.9 | 42.9..42.9 | 1 | 1.443 | - | - | 729.0 | 168.8 | max_abs_diff_vs_ours=0.017546, max_rel_diff_vs_ours=0.009865, mean_nll=5.636059 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/neural.samba-infer.bytes.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 322.2 | 322.2..322.2 | 1 | - | - | - | 230.4 | - | mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 80.1 | 80.1..80.1 | 1 | 4.025 | - | - | 643.9 | - | max_abs_diff_vs_ours=3.666e-06, max_rel_diff_vs_ours=2.061e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 206.4 | 206.4..206.4 | 1 | 1.561 | - | - | 646.8 | - | max_abs_diff_vs_ours=0.016022, max_rel_diff_vs_ours=0.009008, mean_nll=5.635868 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 262.8 | 262.8..262.8 | 1 | 1.226 | - | - | 308.7 | - | max_abs_diff_vs_ours=0.018826, max_rel_diff_vs_ours=0.010584, mean_nll=5.636032 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:13:17Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 82.5 | 82.5..82.5 | 1 | 3.907 | - | - | 310.8 | - | max_abs_diff_vs_ours=2.772e-06, max_rel_diff_vs_ours=1.558e-06, mean_nll=5.635910 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:13:17Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, torch-cpu-compile-fp32, torch-cpu-compile-bf16, torch-cpu-eager-bf16, torch-cpu-eager-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 |
| seed | "none (deterministic)" | 7 | 7 |

### samba-train-step / bytes (neural shape full: B2 L512 DM384 V256 H6 FF1024 layers mamba3+attention+mamba3+attention)

race: done, driver rc 0, log `logs/neural.samba-train-step.bytes.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 669.5 | 669.5..669.5 | 1 | - | - | - | 871.1 | - | loss_first_step=5.635910, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 106.5 | 106.5..106.5 | 1 | 6.286 | - | - | 1933.3 | 1322.8 | loss_first_abs_diff_vs_ours=4.768e-07, loss_first_step=5.635910, loss_last_abs_diff_vs_ours=4.768e-07, loss_last_step=4.833934, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "REFUSED: torch-compile-fp32 on mps failed in round 0 (compile happens here): InductorError('SyntaxError: failed to compile #include <c10/metal/error.h>\\n#include <c10/metal/reduction_utils) (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 107.9 | 107.9..107.9 | 1 | 6.206 | - | - | 1921.6 | 1295.9 | loss_first_abs_diff_vs_ours=9.632e-05, loss_first_step=5.636006, loss_last_abs_diff_vs_ours=0.0001612, loss_last_step=4.833773, steps=2 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
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

race: done, driver rc 0, log `logs/neural.transformer-forward.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 31.1 | 31.1..31.1 | 1 | - | - | - | 522.1 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-eager-fp32 | torch | gpu | opponent | 8.9 | 8.9..8.9 | 1 | 3.503 | - | - | 616.5 | 112.8 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-fp32 | torch | gpu | opponent | 20.3 | 20.3..20.3 | 1 | 1.534 | - | - | 1667.6 | 1064.5 | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.02e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-eager-bf16 | torch | gpu | opponent | 6.0 | 6.0..6.0 | 1 | 5.229 | - | - | 617.4 | 112.8 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-compile-bf16 | torch | gpu | opponent | 4.5 | 4.5..4.5 | 1 | 6.989 | - | - | 646.2 | 40.7 | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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

race: done, driver rc 0, log `logs/neural.transformer-infer.gaussian.shape-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | 66.0 | 66.0..66.0 | 1 | - | - | - | 119.7 | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| torch-cpu-compile-fp32 | torch | cpu | opponent | 5.6 | 5.6..5.6 | 1 | 11.851 | - | - | 364.0 | - | max_abs_diff_vs_ours=9.775e-06, max_rel_diff_vs_ours=2.207e-06 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-compile-bf16 | torch | cpu | opponent | 14.0 | 14.0..14.0 | 1 | 4.704 | - | - | 346.6 | - | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0004107 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |
| torch-cpu-eager-bf16 | torch | cpu | opponent | 17.3 | 17.3..17.3 | 1 | 3.804 | - | - | 265.3 | - | max_abs_diff_vs_ours=0.001941, max_rel_diff_vs_ours=0.0004383 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:00:27Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |
| torch-cpu-eager-fp32 | torch | cpu | opponent | 11.4 | 11.4..11.4 | 1 | 5.784 | - | - | 277.0 | - | max_abs_diff_vs_ours=4.768e-07, max_rel_diff_vs_ours=1.076e-07 | - | LIKE-FOR-LIKE-SPAN | - | ok (stored (measured 2026-09-29T20:00:27Z on ip-172-31-37-211.ec2.internal, cpu (Apple M3 Ultra))) |

memory, ours, torch-cpu-compile-fp32, torch-cpu-compile-bf16, torch-cpu-eager-bf16, torch-cpu-eager-fp32: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 |
| seed | "none (deterministic)" | 7 | 7 |

## Algorithm expansion

### additive-chi2 / istella (rows full, shape X 100000x220; Xq 1000x220; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.additive-chi2.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11.0 | 11.0..11.0 | 1 | - | - | - | 1431.7 | - | kernel_rel_error=0.087730 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 11.0 | 11.0..11.0 | 1 | - | - | - | 1436.8 | - | kernel_rel_error=0.087730 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.8 | 3.8..3.8 | 1 | 2.878 | 2.878 | - | 1264.8 | - | kernel_rel_error=0.087730 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'sample_steps': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.3 | 2.3..2.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.4 | 2.4..2.4 | 1 | 0.920 | 0.965 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### additive-chi2 / taxi (rows full, shape X 100000x11; Xq 1000x11; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.additive-chi2.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.7 | 0.7..0.7 | 1 | - | - | - | 414.7 | - | kernel_rel_error=0.093892 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.6 | 0.6..0.6 | 1 | - | - | - | 415.4 | - | kernel_rel_error=0.093892 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.4 | 0.4..0.4 | 1 | 1.676 | 1.612 | - | 238.0 | - | kernel_rel_error=0.093892 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'sample_steps': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.2 | 1.2..1.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.3 | 0.3..0.3 | 1 | 4.929 | 4.570 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### affinity-prop / istella (rows full, shape X 5000x220; Xq 100000x220; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.affinity-prop.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1847.2 | 1847.2..1847.2 | 1 | - | - | - | 2792.8 | - | n_clusters=342, silhouette=0.089763 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1005.9 | 1005.9..1005.9 | 1 | - | - | - | 2791.0 | - | ari_vs_ours=1.000000, n_clusters=342, silhouette=0.089763 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5734.3 | 5734.3..5734.3 | 1 | 0.322 | 0.175 | - | 1561.5 | - | ari_vs_ours=1.000000, n_clusters=342, silhouette=0.089763 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'affinity': 'euclidean', 'convergence_iter': 15, 'damping': 0.5, 'max_iter': 200, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| affinity | "euclidean" | "euclidean" | "euclidean" |
| max_iter | 200 | 200 | 200 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 52.2 | 52.2..52.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 49.0 | 49.0..49.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 19.0 | 19.0..19.0 | 1 | 2.747 | 2.579 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### affinity-prop / taxi (rows full, shape X 5000x11; Xq 100000x11; y 5000; yq 100000)

race: done, driver rc 0, log `logs/algos.affinity-prop.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1038.3 | 1038.3..1038.3 | 1 | - | - | - | 1911.0 | - | n_clusters=272, silhouette=0.184644 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1020.8 | 1020.8..1020.8 | 1 | - | - | - | 1816.4 | - | ari_vs_ours=1.000000, n_clusters=272, silhouette=0.184644 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4838.6 | 4838.6..4838.6 | 1 | 0.215 | 0.211 | - | 662.5 | - | ari_vs_ours=1.000000, n_clusters=272, silhouette=0.184644 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'affinity': 'euclidean', 'convergence_iter': 15, 'damping': 0.5, 'max_iter': 200, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| affinity | "euclidean" | "euclidean" | "euclidean" |
| max_iter | 200 | 200 | 200 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 4.4 | 4.4..4.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.0 | 4.0..4.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.5 | 5.5..5.5 | 1 | 0.798 | 0.731 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### ard / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.ard.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 53884.0 | 53884.0..53884.0 | 1 | - | - | - | 1716.9 | - | finite=True, r2=-0.122912, rmse=0.885183 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 48109.7 | 48109.7..48109.7 | 1 | - | - | - | 1701.7 | - | finite=True, r2=-0.123501, rmse=0.885416 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10367.4 | 10367.4..10367.4 | 1 | 5.197 | 4.640 | - | 1337.9 | - | finite=True, r2=0.327436, rmse=0.685058 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 14.6 | 14.6..14.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.1 | 15.1..15.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.0 | 4.0..4.0 | 1 | 3.685 | 3.814 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ard / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.ard.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 51.3 | 51.3..51.3 | 1 | - | - | - | 533.5 | - | finite=True, r2=0.909193, rmse=4.799513 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 44.6 | 44.6..44.6 | 1 | - | - | - | 534.0 | - | finite=True, r2=0.909193, rmse=4.799513 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17.3 | 17.3..17.3 | 1 | 2.958 | 2.574 | - | 210.7 | - | finite=True, r2=0.909190, rmse=4.799575 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 0.9 | 0.9..0.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.0 | 1.0..1.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 2.538 | 2.701 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-gmm / istella (rows full, shape X 100000x200; Xq 20000x200; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.bayesian-gmm.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 39953.5 | 39953.5..39953.5 | 1 | - | - | - | 2168.7 | - | mean_log_likelihood=175.322029 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 70783.5 | 70783.5..70783.5 | 1 | - | - | - | 1917.3 | - | mean_log_likelihood=175.561762 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 82775.7 | 82775.7..82775.7 | 1 | 0.483 | 0.855 | - | 1441.8 | - | mean_log_likelihood=201.770716 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'covariance_type': 'full', 'init_params': 'kmeans', 'max_iter': 100, 'n_components': 8, 'n_init': 1, 'random_state': 7, 'reg_covar': 1e-06, 'tol': 0.001, 'weight_concentration_prior_type': 'dirichlet_process'}. Rows: None. Timed: None.

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 55.9 | 55.9..55.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 55.2 | 55.2..55.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 89.6 | 89.6..89.6 | 1 | 0.624 | 0.616 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-gmm / taxi (rows full, shape X 100000x11; Xq 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.bayesian-gmm.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2630.8 | 2630.8..2630.8 | 1 | - | - | - | 449.4 | - | mean_log_likelihood=4.895581 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 444.9 | 444.9..444.9 | 1 | - | - | - | 424.3 | - | mean_log_likelihood=4.895586 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4017.7 | 4017.7..4017.7 | 1 | 0.655 | 0.111 | - | 258.3 | - | mean_log_likelihood=6.178321 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'covariance_type': 'full', 'init_params': 'kmeans', 'max_iter': 100, 'n_components': 8, 'n_init': 1, 'random_state': 7, 'reg_covar': 1e-06, 'tol': 0.001, 'weight_concentration_prior_type': 'dirichlet_process'}. Rows: None. Timed: None.

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

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 13.7 | 13.7..13.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 13.6 | 13.6..13.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 5.9 | 5.9..5.9 | 1 | 2.302 | 2.289 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-ridge / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bayesian-ridge.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 101826.8 | 101826.8..101826.8 | 1 | - | - | - | 2365.5 | - | finite=True, r2=-41643.670747, rmse=170.466932 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 92402.6 | 92402.6..92402.6 | 1 | - | - | - | 2389.7 | - | finite=True, r2=-41688.617698, rmse=170.558899 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6789.4 | 6789.4..6789.4 | 1 | 14.998 | 13.610 | - | 4654.0 | - | finite=True, r2=-890.186917, rmse=24.937036 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 23.5 | 23.5..23.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.8 | 15.8..15.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.9 | 3.9..3.9 | 1 | 6.099 | 4.088 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bayesian-ridge / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bayesian-ridge.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 482.5 | 482.5..482.5 | 1 | - | - | - | 588.0 | - | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 435.5 | 435.5..435.5 | 1 | - | - | - | 585.4 | - | finite=True, r2=0.908981, rmse=4.805108 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 70.8 | 70.8..70.8 | 1 | 6.812 | 6.149 | - | 383.2 | - | finite=True, r2=0.908983, rmse=4.805052 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1.4 | 1.4..1.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.5 | 0.5..0.5 | 1 | 2.858 | 2.597 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### bisecting-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bisecting-kmeans.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3995.9 | 3995.9..3995.9 | 1 | - | - | - | 9245.5 | - | n_clusters=8, silhouette=0.118345 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3735.0 | 3735.0..3735.0 | 1 | - | - | - | 9246.2 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.118345 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1427.6 | 1427.6..1427.6 | 1 | 2.799 | 2.616 | - | 3074.4 | - | ari_vs_ours=0.692929, n_clusters=8, silhouette=0.096414 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 22.3 | 22.3..22.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 41.8 | 41.8..41.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 60.3 | 60.3..60.3 | 1 | 0.369 | 0.694 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### bisecting-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.bisecting-kmeans.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 403.2 | 403.2..403.2 | 1 | - | - | - | 937.1 | - | n_clusters=8, silhouette=0.155357 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 396.9 | 396.9..396.9 | 1 | - | - | - | 901.2 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.155357 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 383.4 | 383.4..383.4 | 1 | 1.052 | 1.035 | - | 349.6 | - | ari_vs_ours=0.502256, n_clusters=8, silhouette=0.128033 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.0 | 2.0..2.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 15.6 | 15.6..15.6 | 1 | 0.151 | 0.130 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### enet-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.enet-cv.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 80217.1 | 80217.1..80217.1 | 1 | - | - | - | 2379.6 | - | finite=True, r2=0.316583, rmse=0.690563 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 74002.5 | 74002.5..74002.5 | 1 | - | - | - | 2430.7 | - | finite=True, r2=0.316583, rmse=0.690563 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7103.7 | 7103.7..7103.7 | 1 | 11.292 | 10.418 | - | 8037.5 | - | finite=True, r2=0.317292, rmse=0.690205 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 17.2 | 17.2..17.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.4 | 16.4..16.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.0 | 4.0..4.0 | 1 | 4.313 | 4.118 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### enet-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.enet-cv.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3259.6 | 3259.6..3259.6 | 1 | - | - | - | 627.4 | - | finite=True, r2=0.909002, rmse=4.804540 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3164.4 | 3164.4..3164.4 | 1 | - | - | - | 626.0 | - | finite=True, r2=0.909002, rmse=4.804540 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 212.6 | 212.6..212.6 | 1 | 15.329 | 14.882 | - | 496.5 | - | finite=True, r2=0.909004, rmse=4.804486 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 6.238 | 6.285 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### huber / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.huber.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 46713.1 | 46713.1..46713.1 | 1 | - | - | - | 3191.1 | - | finite=True, r2=-0.009166, rmse=0.839154 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 48657.4 | 48657.4..48657.4 | 1 | - | - | - | 3104.3 | - | finite=True, r2=-0.008227, rmse=0.838764 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 45695.1 | 45695.1..45695.1 | 1 | 1.022 | 1.065 | - | 3838.2 | - | finite=True, r2=-0.010176, rmse=0.839574 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 8.0 | 8.0..8.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 10.9 | 10.9..10.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 16.0 | 16.0..16.0 | 1 | 0.501 | 0.679 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### huber / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.huber.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20061.9 | 20061.9..20061.9 | 1 | - | - | - | 710.4 | - | finite=True, r2=0.900215, rmse=5.031176 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 165697.8 | 165697.8..165697.8 | 1 | - | - | - | 893.3 | - | finite=True, r2=0.900217, rmse=5.031119 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1851.4 | 1851.4..1851.4 | 1 | 10.836 | 89.498 | - | 471.0 | - | finite=True, r2=0.900215, rmse=5.031163 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.8 | 1.8..1.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.1 | 1.1..1.1 | 1 | 1.527 | 1.550 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### isotonic / istella (rows full, shape X 1000000; Xq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.isotonic.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 941.8 | 941.8..941.8 | 1 | - | - | - | 1694.4 | - | finite=True, r2=0.187985, rmse=0.752735 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 900.3 | 900.3..900.3 | 1 | - | - | - | 1702.7 | - | finite=True, r2=0.187985, rmse=0.752735 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 37.5 | 37.5..37.5 | 1 | 25.145 | 24.038 | - | 1126.4 | - | finite=True, r2=0.187985, rmse=0.752735 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 6.9 | 6.9..6.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 7.0 | 7.0..7.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.3 | 2.3..2.3 | 1 | 3.051 | 3.099 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### isotonic / taxi (rows full, shape X 1000000; Xq 100000; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.isotonic.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1404.8 | 1404.8..1404.8 | 1 | - | - | - | 902.6 | - | finite=True, r2=0.897069, rmse=5.109874 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1342.7 | 1342.7..1342.7 | 1 | - | - | - | 902.9 | - | finite=True, r2=0.897069, rmse=5.109874 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 90.5 | 90.5..90.5 | 1 | 15.528 | 14.841 | - | 334.7 | - | finite=True, r2=0.897069, rmse=5.109874 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 9.1 | 9.1..9.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 9.3 | 9.3..9.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.6 | 4.6..4.6 | 1 | 1.989 | 2.034 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### knn-imputer / taxi (rows full, shape X 100000x11; X_true 100000x11; Xq 20000x11; Xq_true 20000x11; y 100000; yq 20000)

race: done, driver rc 0, log `logs/algos.knn-imputer.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 60.4 | 60.4..60.4 | 1 | - | - | - | 414.8 | - | masked_rmse=6.151696, max_abs_diff_vs_sklearn=29.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 59.9 | 59.9..59.9 | 1 | - | - | - | 425.5 | - | masked_rmse=6.151696, max_abs_diff_vs_sklearn=29.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.7 | 1.7..1.7 | 1 | 36.051 | 35.761 | - | 1303.8 | - | masked_rmse=5.256719 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| metric | "nan_euclidean" | "nan_euclidean" | "nan_euclidean" |
| n_neighbors | 5 | 5 | 5 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| weights | "uniform" | "uniform" | "uniform" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 317.4 | 317.4..317.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 338.2 | 338.2..338.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 25457.6 | 25457.6..25457.6 | 1 | 0.012 | 0.013 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### label-propagation / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; y_semi 200000; yq 20000)

race: failed, driver rc 0, log `logs/algos.label-propagation.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 234.5 GB: the driver's process tree held 255.6 GB, over 90% of the box's 274.9 GB) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 229.9 GB: the driver's process tree held 249.4 GB, over 90% of the box's 274.9 GB) |
| sklearn-cpu | scikit-learn | cpu | opponent | 14551.5 | 14551.5..14551.5 | 1 | - | - | - | 1368.5 | - | accuracy=0.905500 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'gamma': 20, 'kernel': 'knn', 'max_iter': 1000, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 1000 | 1000 | 1000 |
| n_neighbors | 7 | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 1268.4 | 1268.4..1268.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### label-propagation / taxi (rows full, shape X 200000x11; Xq 20000x11; y 200000; y_semi 200000; yq 20000)

race: failed, driver rc 0, log `logs/algos.label-propagation.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 232.1 GB: the driver's process tree held 249.6 GB, over 90% of the box's 274.9 GB) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 243.9 GB: the driver's process tree held 262.2 GB, over 90% of the box's 274.9 GB) |
| sklearn-cpu | scikit-learn | cpu | opponent | 10366.0 | 10366.0..10366.0 | 1 | - | - | - | 328.6 | - | accuracy=0.701600 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'gamma': 20, 'kernel': 'knn', 'max_iter': 1000, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 1000 | 1000 | 1000 |
| n_neighbors | 7 | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 1004.6 | 1004.6..1004.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### label-spreading / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; y_semi 200000; yq 20000)

race: failed, driver rc 0, log `logs/algos.label-spreading.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 240.5 GB: the driver's process tree held 262.0 GB, over 90% of the box's 274.9 GB) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 242.0 GB: the driver's process tree held 262.4 GB, over 90% of the box's 274.9 GB) |
| sklearn-cpu | scikit-learn | cpu | opponent | 10273.3 | 10273.3..10273.3 | 1 | - | - | - | 1395.3 | - | accuracy=0.904450 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.2, 'gamma': 20, 'kernel': 'knn', 'max_iter': 30, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.2 | 0.2 | 0.2 |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 30 | 30 | 30 |
| n_neighbors | 7 | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 1277.1 | 1277.1..1277.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### label-spreading / taxi (rows full, shape X 200000x11; Xq 20000x11; y 200000; y_semi 200000; yq 20000)

race: failed, driver rc 0, log `logs/algos.label-spreading.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 243.8 GB: the driver's process tree held 262.3 GB, over 90% of the box's 274.9 GB) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | HOST-MEMORY(killed at 233.0 GB: the driver's process tree held 250.5 GB, over 90% of the box's 274.9 GB) |
| sklearn-cpu | scikit-learn | cpu | opponent | 6647.9 | 6647.9..6647.9 | 1 | - | - | - | 360.0 | - | accuracy=0.676400 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.2, 'gamma': 20, 'kernel': 'knn', 'max_iter': 30, 'n_neighbors': 7, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.2 | 0.2 | 0.2 |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 30 | 30 | 30 |
| n_neighbors | 7 | 7 | 7 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| sklearn-cpu | Xq | - | 959.8 | 959.8..959.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11639.6 | 11639.6..11639.6 | 1 | - | - | - | 2412.7 | - | finite=True, r2=-1.360350, rmse=1.283360 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10382.8 | 10382.8..10382.8 | 1 | - | - | - | 2395.6 | - | finite=True, r2=-1.632561, rmse=1.355344 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
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
| mojolearn IDENTICAL | Xq | - | 16.4 | 16.4..16.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.0 | 16.0..16.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(error: {"error": "OverflowError('int too large to convert to float')", "event": "error", "stage": "round 0"}) |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lars.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 171.0 | 171.0..171.0 | 1 | - | - | - | 587.6 | - | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 155.6 | 155.6..155.6 | 1 | - | - | - | 584.1 | - | finite=True, r2=0.908981, rmse=4.805109 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 35.8 | 35.8..35.8 | 1 | 4.778 | 4.348 | - | 244.1 | - | finite=True, r2=0.908988, rmse=4.804917 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1.1 | 1.1..1.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.0 | 1.0..1.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.3 | 0.3..0.3 | 1 | 3.320 | 3.033 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-cv.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 76137.4 | 76137.4..76137.4 | 1 | - | - | - | 2480.9 | - | finite=True, r2=0.310329, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 70153.0 | 70153.0..70153.0 | 1 | - | - | - | 2478.2 | - | finite=True, r2=0.310329, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5538.1 | 5538.1..5538.1 | 1 | 13.748 | 12.667 | - | 8020.4 | - | finite=True, r2=0.310837, rmse=0.693460 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 17.3 | 17.3..17.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 17.4 | 17.4..17.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.9 | 3.9..3.9 | 1 | 4.477 | 4.497 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-cv.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3251.7 | 3251.7..3251.7 | 1 | - | - | - | 632.9 | - | finite=True, r2=0.909059, rmse=4.803051 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3172.1 | 3172.1..3172.1 | 1 | - | - | - | 626.0 | - | finite=True, r2=0.909059, rmse=4.803051 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 143.2 | 143.2..143.2 | 1 | 22.708 | 22.152 | - | 497.8 | - | finite=True, r2=0.909038, rmse=4.803593 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 5.898 | 5.673 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-lars / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-lars.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8933.8 | 8933.8..8933.8 | 1 | - | - | - | 2408.5 | - | finite=True, r2=0.310330, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7872.1 | 7872.1..7872.1 | 1 | - | - | - | 2375.3 | - | finite=True, r2=0.310330, rmse=0.693715 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 229.4 | 229.4..229.4 | 1 | 38.947 | 34.318 | - | 1919.7 | - | finite=True, r2=0.311104, rmse=0.693326 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 15.0 | 15.0..15.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.0 | 15.0..15.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.9 | 3.9..3.9 | 1 | 3.873 | 3.878 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lasso-lars / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.lasso-lars.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 172.2 | 172.2..172.2 | 1 | - | - | - | 583.5 | - | finite=True, r2=0.908996, rmse=4.804699 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 148.7 | 148.7..148.7 | 1 | - | - | - | 581.8 | - | finite=True, r2=0.908996, rmse=4.804698 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34.7 | 34.7..34.7 | 1 | 4.962 | 4.285 | - | 242.6 | - | finite=True, r2=0.909003, rmse=4.804527 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 1.3 | 1.3..1.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.2 | 1.2..1.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | 2.962 | 2.804 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### lof / istella (rows full, shape X 200000x220; Xq 100000x220; y 200000; yq 100000)

race: done, driver rc 0, log `logs/algos.lof.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 328024.7 | 328024.7..328024.7 | 1 | - | - | - | 1934.2 | - | fraction_flagged=0.033610, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 295424.6 | 295424.6..295424.6 | 1 | - | - | - | 1948.1 | - | fraction_flagged=0.033610, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10177.4 | 10177.4..10177.4 | 1 | 32.231 | 29.028 | - | 1503.8 | - | fraction_flagged=0.033610, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'brute', 'contamination': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'n_neighbors': 20, 'novelty': False, 'p': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| contamination | "auto" | "auto" | "auto" |
| leaf_size | 30 | 30 | 30 |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_neighbors | 20 | 20 | 20 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### lof / taxi (rows full, shape X 200000x11; Xq 100000x11; y 200000; yq 100000)

race: done, driver rc 0, log `logs/algos.lof.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2172.9 | 2172.9..2172.9 | 1 | - | - | - | 545.2 | - | fraction_flagged=0.008960, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2027.3 | 2027.3..2027.3 | 1 | - | - | - | 529.0 | - | fraction_flagged=0.008960, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3405.3 | 3405.3..3405.3 | 1 | 0.638 | 0.595 | - | 425.3 | - | fraction_flagged=0.008960, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'brute', 'contamination': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'n_neighbors': 20, 'novelty': False, 'p': 2}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| contamination | "auto" | "auto" | "auto" |
| leaf_size | 30 | 30 | 30 |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_neighbors | 20 | 20 | 20 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

### logreg-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.logreg-cv.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 732287.2 | 732287.2..732287.2 | 1 | - | - | - | 2513.1 | - | accuracy=0.924520, logloss=0.181412 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 72840.3 | 72840.3..72840.3 | 1 | - | 10.053 | - | 2981.6 | - | accuracy=0.924630, logloss=0.181351 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'Cs': [0.1, 1.0, 10.0], 'cv': 5, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1.0, 'max_iter': 1000, 'penalty': 'l2', 'random_state': 7, 'refit': True, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| class_weight | null | null | null |
| cv | 5 | 5 | 5 |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | - | - | 0 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | REFUSED(timeout: null) |
| mojolearn FAST | Xq | - | 57.2 | 57.2..57.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 24.3 | 24.3..24.3 | 1 | - | 2.354 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### meanshift / istella (rows full, shape X 10000x220; Xq 100000x220; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.meanshift.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6424.7 | 6424.7..6424.7 | 1 | - | - | - | 1455.5 | - | n_clusters=12, silhouette=0.403452 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5618.9 | 5618.9..5618.9 | 1 | - | - | - | 1454.9 | - | ari_vs_ours=1.000000, n_clusters=12, silhouette=0.403452 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 444.3 | 444.3..444.3 | 1 | 14.460 | 12.646 | - | 1167.4 | - | ari_vs_ours=1.000000, n_clusters=12, silhouette=0.403452 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 19.5 | 19.5..19.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 20.0 | 20.0..20.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 11.8 | 11.8..11.8 | 1 | 1.653 | 1.694 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### meanshift / taxi (rows full, shape X 10000x11; Xq 100000x11; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.meanshift.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 660.2 | 660.2..660.2 | 1 | - | - | - | 388.4 | - | n_clusters=122, silhouette=0.246631 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 549.2 | 549.2..549.2 | 1 | - | - | - | 390.2 | - | ari_vs_ours=0.999996, n_clusters=122, silhouette=0.246592 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9370.4 | 9370.4..9370.4 | 1 | 0.070 | 0.059 | - | 216.5 | - | ari_vs_ours=1.000000, n_clusters=122, silhouette=0.246631 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.5 | 2.5..2.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.5 | 2.5..2.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 18.7 | 18.7..18.7 | 1 | 0.132 | 0.133 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### minibatch-kmeans / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minibatch-kmeans.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 813.6 | 813.6..813.6 | 1 | - | - | - | 3103.5 | - | n_clusters=8, silhouette=0.116696 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 799.4 | 799.4..799.4 | 1 | - | - | - | 3109.0 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.116696 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 123.5 | 123.5..123.5 | 1 | 6.590 | 6.475 | - | 1132.5 | - | ari_vs_ours=0.622446, n_clusters=8, silhouette=0.111849 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 17.8 | 17.8..17.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 17.4 | 17.4..17.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 6.1 | 6.1..6.1 | 1 | 2.913 | 2.851 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### minibatch-kmeans / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.minibatch-kmeans.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 196.1 | 196.1..196.1 | 1 | - | - | - | 514.1 | - | n_clusters=8, silhouette=0.138060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 195.3 | 195.3..195.3 | 1 | - | - | - | 511.1 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.138060 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 43.9 | 43.9..43.9 | 1 | 4.464 | 4.446 | - | 227.1 | - | ari_vs_ours=0.525151, n_clusters=8, silhouette=0.165473 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 2.4 | 2.4..2.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.5 | 1.5..1.5 | 1 | 1.550 | 1.220 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### nearest-centroid / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nearest-centroid.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5529.0 | 5529.0..5529.0 | 1 | - | - | - | 2214.3 | - | accuracy=0.852610, logloss=4.299221 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2441.6 | 2441.6..2441.6 | 1 | - | - | - | 2197.2 | - | accuracy=0.852610, logloss=4.299222 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 529.4 | 529.4..529.4 | 1 | 10.444 | 4.612 | - | 4710.9 | - | accuracy=0.852610, logloss=4.117692 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| metric | "euclidean" | "euclidean" | "euclidean" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 39.8 | 39.8..39.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 40.9 | 40.9..40.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 169.4 | 169.4..169.4 | 1 | 0.235 | 0.241 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### nearest-centroid / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.nearest-centroid.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 505.6 | 505.6..505.6 | 1 | - | - | - | 450.6 | - | accuracy=0.666750, logloss=0.782162 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 422.0 | 422.0..422.0 | 1 | - | - | - | 428.5 | - | accuracy=0.666750, logloss=0.782162 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 96.9 | 96.9..96.9 | 1 | 5.219 | 4.357 | - | 451.3 | - | accuracy=0.666750, logloss=0.781690 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| metric | "euclidean" | "euclidean" | "euclidean" |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 25.3 | 25.3..25.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 25.4 | 25.4..25.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 12.9 | 12.9..12.9 | 1 | 1.961 | 1.971 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ocsvm / istella (rows full, shape X 10000x220; Xq 10000x220; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.ocsvm.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 301.6 | 301.6..301.6 | 1 | - | - | - | 2114.5 | - | fraction_flagged=0.078300, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 290.4 | 290.4..290.4 | 1 | - | - | - | 2114.5 | - | fraction_flagged=0.078300, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 691.0 | 691.0..691.0 | 1 | 0.436 | 0.420 | - | 1202.5 | - | fraction_flagged=0.078300, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'coef0': 0.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'nu': 0.1, 'shrinking': True, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 12.9 | 12.9..12.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 15.1 | 15.1..15.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 754.9 | 754.9..754.9 | 1 | 0.017 | 0.020 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ocsvm / taxi (rows full, shape X 10000x11; Xq 10000x11; y 10000; yq 10000)

race: done, driver rc 0, log `logs/algos.ocsvm.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 212.3 | 212.3..212.3 | 1 | - | - | - | 1203.8 | - | fraction_flagged=0.136100, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 220.7 | 220.7..220.7 | 1 | - | - | - | 1203.5 | - | fraction_flagged=0.136100, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 180.7 | 180.7..180.7 | 1 | 1.175 | 1.221 | - | 287.6 | - | fraction_flagged=0.136100, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'coef0': 0.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'nu': 0.1, 'shrinking': True, 'tol': 0.001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| tol | 0.001 | 0.001 | 0.001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 7.1 | 7.1..7.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 4.5 | 4.5..4.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 222.0 | 222.0..222.0 | 1 | 0.032 | 0.020 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### optics / istella (rows full, shape X 10000x220; Xq 100000x220; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.optics.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 580.3 | 580.3..580.3 | 1 | - | - | - | 2041.8 | - | n_clusters=20, silhouette=-0.287356 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 562.4 | 562.4..562.4 | 1 | - | - | - | 2045.1 | - | ari_vs_ours=1.000000, n_clusters=20, silhouette=-0.287356 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 30143.0 | 30143.0..30143.0 | 1 | 0.019 | 0.019 | - | 1140.8 | - | ari_vs_ours=0.984603, n_clusters=20, silhouette=-0.285806 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'auto', 'cluster_method': 'xi', 'leaf_size': 30, 'max_eps': inf, 'metric': 'minkowski', 'min_samples': 10, 'p': 2, 'predecessor_correction': True, 'xi': 0.05}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| eps | null | null | null |
| leaf_size | 30 | 30 | 30 |
| metric | "minkowski" | "minkowski" | "minkowski" |
| min_cluster_size | null | null | null |
| min_samples | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast eps: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast min_cluster_size: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu eps: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu min_cluster_size: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.0 | 0.0..0.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 0.0 | 0.0..0.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.0 | 0.0..0.0 | 1 | 0.272 | 0.466 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### optics / taxi (rows full, shape X 10000x11; Xq 100000x11; y 10000; yq 100000)

race: done, driver rc 0, log `logs/algos.optics.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 451.4 | 451.4..451.4 | 1 | - | - | - | 1143.0 | - | n_clusters=127, silhouette=-0.353359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 439.9 | 439.9..439.9 | 1 | - | - | - | 1142.8 | - | ari_vs_ours=1.000000, n_clusters=127, silhouette=-0.353359 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 767657.4 | 767657.4..767657.4 | 1 | 0.0005881 | 0.0005731 | - | 224.8 | - | ari_vs_ours=1.000000, n_clusters=127, silhouette=-0.353359 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'auto', 'cluster_method': 'xi', 'leaf_size': 30, 'max_eps': inf, 'metric': 'minkowski', 'min_samples': 10, 'p': 2, 'predecessor_correction': True, 'xi': 0.05}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| eps | null | null | null |
| leaf_size | 30 | 30 | 30 |
| metric | "minkowski" | "minkowski" | "minkowski" |
| min_cluster_size | null | null | null |
| min_samples | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

accepted difference: ours-fast eps: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast min_cluster_size: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu eps: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu min_cluster_size: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 0.0 | 0.0..0.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 0.0 | 0.0..0.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.0 | 0.0..0.0 | 1 | 0.500 | 0.771 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq) or fit labels(Xq)

inference call, ours-fast: predict(Xq) or fit labels(Xq)

inference call, sklearn-cpu: predict(Xq) or fit labels(Xq)

### pa-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-clf.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 249929.0 | 249929.0..249929.0 | 1 | - | - | - | 2385.6 | - | accuracy=0.903700 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 200552.1 | 200552.1..200552.1 | 1 | - | - | - | 2385.1 | - | accuracy=0.903700 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 9373.5 | 9373.5..9373.5 | 1 | 26.663 | 21.396 | - | 1088.0 | - | accuracy=0.890480 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 24.6 | 24.6..24.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 24.4 | 24.4..24.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.6 | 4.6..4.6 | 1 | 5.367 | 5.315 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-clf.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 23970.4 | 23970.4..23970.4 | 1 | - | - | - | 691.0 | - | accuracy=0.583830 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 39286.1 | 39286.1..39286.1 | 1 | - | - | - | 659.3 | - | accuracy=0.583830 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1215.9 | 1215.9..1215.9 | 1 | 19.713 | 32.309 | - | 204.4 | - | accuracy=0.744740 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 15.9 | 15.9..15.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.4 | 16.4..16.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.0 | 1.0..1.0 | 1 | 16.187 | 16.654 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-reg.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 254962.9 | 254962.9..254962.9 | 1 | - | - | - | 2363.1 | - | finite=True, r2=-0.282312, rmse=0.945926 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 198976.9 | 198976.9..198976.9 | 1 | - | - | - | 2367.4 | - | finite=True, r2=-0.282312, rmse=0.945926 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12373.6 | 12373.6..12373.6 | 1 | 20.605 | 16.081 | - | 1084.9 | - | finite=True, r2=-0.128155, rmse=0.887248 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 21.2 | 21.2..21.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 22.2 | 22.2..22.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.1 | 4.1..4.1 | 1 | 5.175 | 5.421 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### pa-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.pa-reg.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24670.9 | 24670.9..24670.9 | 1 | - | - | - | 662.8 | - | finite=True, r2=0.853920, rmse=6.087406 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 41674.2 | 41674.2..41674.2 | 1 | - | - | - | 600.6 | - | finite=True, r2=0.853920, rmse=6.087406 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1199.1 | 1199.1..1199.1 | 1 | 20.574 | 34.754 | - | 206.2 | - | finite=True, r2=0.795107, rmse=7.209421 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 12.4 | 12.4..12.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.7 | 8.7..8.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.5 | 0.5..0.5 | 1 | 24.039 | 16.853 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### perceptron / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.perceptron.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 160515.6 | 160515.6..160515.6 | 1 | - | - | - | 2310.3 | - | accuracy=0.882480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 134524.4 | 134524.4..134524.4 | 1 | - | - | - | 2384.3 | - | accuracy=0.882480 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5724.9 | 5724.9..5724.9 | 1 | 28.038 | 23.498 | - | 1087.9 | - | accuracy=0.896130 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 23.6 | 23.6..23.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 24.5 | 24.5..24.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.4 | 4.4..4.4 | 1 | 5.303 | 5.515 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### perceptron / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.perceptron.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 23038.7 | 23038.7..23038.7 | 1 | - | - | - | 684.0 | - | accuracy=0.465380 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 40319.4 | 40319.4..40319.4 | 1 | - | - | - | 670.4 | - | accuracy=0.465380 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 907.3 | 907.3..907.3 | 1 | 25.391 | 44.437 | - | 200.3 | - | accuracy=0.750520 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 15.9 | 15.9..15.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.0 | 16.0..16.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.0 | 1.0..1.0 | 1 | 15.861 | 15.957 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### poisson / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.poisson.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 55094.7 | 55094.7..55094.7 | 1 | - | - | - | 667.8 | - | finite=True, r2=0.035965, rmse=15.638069 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 42268.4 | 42268.4..42268.4 | 1 | - | - | - | 643.8 | - | finite=True, r2=0.035965, rmse=15.638068 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 401.3 | 401.3..401.3 | 1 | 137.286 | 105.325 | - | 314.4 | - | finite=True, r2=0.036205, rmse=15.636127 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.2 | 8.2..8.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.3 | 8.3..8.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 3.1 | 3.1..3.1 | 1 | 2.609 | 2.647 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### poly-count-sketch / istella (rows full, shape X 100000x220; Xq 1000x220; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.poly-count-sketch.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.4 | 0.4..0.4 | 1 | - | - | - | 1345.3 | - | kernel_rel_error=0.040849 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.4 | 0.4..0.4 | 1 | - | - | - | 1348.9 | - | kernel_rel_error=0.040849 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.1 | 3.1..3.1 | 1 | 0.120 | 0.116 | - | 1181.2 | - | kernel_rel_error=0.040849 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'coef0': 0, 'degree': 2, 'gamma': 1.0, 'n_components': 256, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0 | 0 | 0 |
| degree | 2 | 2 | 2 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.2 | 2.2..2.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.9 | 4.9..4.9 | 1 | 0.454 | 0.428 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### poly-count-sketch / taxi (rows full, shape X 100000x11; Xq 1000x11; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.poly-count-sketch.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 388.6 | - | kernel_rel_error=0.096596 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.3 | 0.3..0.3 | 1 | - | - | - | 387.5 | - | kernel_rel_error=0.096596 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.5 | 0.5..0.5 | 1 | 0.543 | 0.486 | - | 211.1 | - | kernel_rel_error=0.096596 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'coef0': 0, 'degree': 2, 'gamma': 1.0, 'n_components': 256, 'random_state': 7}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0 | 0 | 0 |
| degree | 2 | 2 | 2 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 1.9 | 1.9..1.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 1.7 | 1.7..1.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 2.7 | 2.7..2.7 | 1 | 0.702 | 0.621 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### quantile / istella (rows full, shape X 100000x220; Xq 100000x220; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 235712.1 | 235712.1..235712.1 | 1 | - | - | - | 1734.7 | - | finite=False, r2=nan, rmse=nan | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 211403.8 | 211403.8..211403.8 | 1 | - | - | - | 1736.1 | - | finite=False, r2=nan, rmse=nan | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 874304.1 | 874304.1..874304.1 | 1 | 0.270 | 0.242 | - | 3515.8 | - | finite=True, r2=-0.044780, rmse=0.853833 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 18.5 | 18.5..18.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 14.7 | 14.7..14.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 16.2 | 16.2..16.2 | 1 | 1.140 | 0.905 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### quantile / taxi (rows full, shape X 100000x11; Xq 100000x11; y 100000; yq 100000)

race: done, driver rc 0, log `logs/algos.quantile.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 30505.1 | 30505.1..30505.1 | 1 | - | - | - | 613.1 | - | finite=True, r2=0.900039, rmse=5.035615 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 24477.0 | 24477.0..24477.0 | 1 | - | - | - | 604.3 | - | finite=True, r2=0.899875, rmse=5.039731 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 216313.7 | 216313.7..216313.7 | 1 | 0.141 | 0.113 | - | 464.2 | - | finite=True, r2=0.899678, rmse=5.044706 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 8.1 | 8.1..8.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 8.3 | 8.3..8.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.3 | 1.3..1.3 | 1 | 6.149 | 6.277 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### radius-neighbors / istella (rows full, shape X 200000x220; Xq 20000x220; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.radius-neighbors.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 2204.5 | - | count_agreement_vs_sklearn=1.000000, neighbors_total=1220718 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.3 | 0.3..0.3 | 1 | - | - | - | 2194.8 | - | count_agreement_vs_sklearn=1.000000, neighbors_total=1220718 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5.7 | 5.7..5.7 | 1 | 0.048 | 0.050 | - | 1402.9 | - | neighbors_total=1220718 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'auto', 'metric': 'euclidean', 'p': 2}. Rows: None. Timed: None.

mismatch: algorithm='auto' on both: ours' auto is its random ball cover (exact, triangle-inequality pruning), scikit-learn's picks a KD/ball tree or brute force; both return the exact neighbour set

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | - | - | 5 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3129.9 | 3129.9..3129.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2991.3 | 2991.3..2991.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1216.1 | 1216.1..1216.1 | 1 | 2.574 | 2.460 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: radius_neighbors(Xq)(Xq)

inference call, ours-fast: radius_neighbors(Xq)(Xq)

inference call, sklearn-cpu: radius_neighbors(Xq)(Xq)

### radius-neighbors / taxi (rows full, shape X 200000x11; Xq 20000x11; y 200000; yq 20000)

race: done, driver rc 0, log `logs/algos.radius-neighbors.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.1 | 0.1..0.1 | 1 | - | - | - | 457.3 | - | count_agreement_vs_sklearn=1.000000, neighbors_total=31 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.1 | 0.1..0.1 | 1 | - | - | - | 454.8 | - | count_agreement_vs_sklearn=1.000000, neighbors_total=31 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 45.0 | 45.0..45.0 | 1 | 0.003 | 0.003 | - | 274.0 | - | neighbors_total=31 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'auto', 'metric': 'euclidean', 'p': 2}. Rows: None. Timed: None.

mismatch: algorithm='auto' on both: ours' auto is its random ball cover (exact, triangle-inequality pruning), scikit-learn's picks a KD/ball tree or brute force; both return the exact neighbour set

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | - | - | 5 |
| p | 2 | 2 | 2 |
| seed | "none (deterministic)" | "none (deterministic)" | "none (deterministic)" |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 164.1 | 164.1..164.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 151.9 | 151.9..151.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 35.8 | 35.8..35.8 | 1 | 4.580 | 4.242 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: radius_neighbors(Xq)(Xq)

inference call, ours-fast: radius_neighbors(Xq)(Xq)

inference call, sklearn-cpu: radius_neighbors(Xq)(Xq)

### ridge-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-clf.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8393.7 | 8393.7..8393.7 | 1 | - | - | - | 2429.6 | - | accuracy=0.894330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9771.2 | 9771.2..9771.2 | 1 | - | - | - | 2361.9 | - | accuracy=0.894330 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6439.7 | 6439.7..6439.7 | 1 | 1.303 | 1.517 | - | 9500.9 | - | accuracy=0.910540 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 21.9 | 21.9..21.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 22.4 | 22.4..22.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.4 | 4.4..4.4 | 1 | 4.986 | 5.095 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-clf.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 384.6 | 384.6..384.6 | 1 | - | - | - | 645.5 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 396.5 | 396.5..396.5 | 1 | - | - | - | 628.4 | - | accuracy=0.763570 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 74.5 | 74.5..74.5 | 1 | 5.161 | 5.321 | - | 328.7 | - | accuracy=0.763580 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 9.7 | 9.7..9.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 9.3 | 9.3..9.3 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.8 | 0.8..0.8 | 1 | 12.014 | 11.535 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-cv / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-cv.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 127082.4 | 127082.4..127082.4 | 1 | - | - | - | 8807.9 | - | finite=True, r2=0.328683, rmse=0.684423 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 3.9 | 3.9..3.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### ridge-cv / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.ridge-cv.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "ValueError('mojolearn RidgeCV: only cv=None, scoring=None, alpha_per_target=False are implemented')", "event": "error", "stage": "round 0"}) |
| sklearn-cpu | scikit-learn | cpu | opponent | 952.2 | 952.2..952.2 | 1 | - | - | - | 335.8 | - | finite=True, r2=0.908988, rmse=4.804917 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| sklearn-cpu | Xq | - | 0.4 | 0.4..0.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-clf / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 835179.1 | 835179.1..835179.1 | 1 | - | - | - | 2392.9 | - | accuracy=0.901150 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 753448.6 | 753448.6..753448.6 | 1 | - | - | - | 2388.5 | - | accuracy=0.901150 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 29000.6 | 29000.6..29000.6 | 1 | 28.799 | 25.980 | - | 1086.6 | - | accuracy=0.910200 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
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
| mojolearn IDENTICAL | Xq | - | 23.2 | 23.2..23.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 23.8 | 23.8..23.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.6 | 4.6..4.6 | 1 | 5.091 | 5.226 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-clf / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-clf.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 115209.3 | 115209.3..115209.3 | 1 | - | - | - | 638.3 | - | accuracy=0.766410 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 195485.5 | 195485.5..195485.5 | 1 | - | - | - | 668.9 | - | accuracy=0.766410 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5272.5 | 5272.5..5272.5 | 1 | 21.851 | 37.077 | - | 206.1 | - | accuracy=0.752520 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs=100, the others max_iter=100

config: cuML benchmark (RAPIDS), MBSGDClassifier (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
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
| mojolearn IDENTICAL | Xq | - | 14.1 | 14.1..14.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 16.1 | 16.1..16.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.9 | 0.9..0.9 | 1 | 14.893 | 17.054 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-ocsvm / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 166821.3 | 166821.3..166821.3 | 1 | - | - | - | 2279.1 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 150867.3 | 150867.3..150867.3 | 1 | - | - | - | 2352.5 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5739.9 | 5739.9..5739.9 | 1 | 29.063 | 26.284 | - | 1088.8 | - | fraction_flagged=0.093340, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 23.0 | 23.0..23.0 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 22.6 | 22.6..22.6 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.3 | 4.3..4.3 | 1 | 5.397 | 5.307 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-ocsvm / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 23428.7 | 23428.7..23428.7 | 1 | - | - | - | 652.3 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 39435.7 | 39435.7..39435.7 | 1 | - | - | - | 636.7 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 900.7 | 900.7..900.7 | 1 | 26.012 | 43.783 | - | 209.0 | - | fraction_flagged=0.007020, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

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
| mojolearn IDENTICAL | Xq | - | 17.4 | 17.4..17.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 17.7 | 17.7..17.7 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.6 | 0.6..0.6 | 1 | 29.499 | 29.895 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-reg / istella (rows full, shape X 1000000x220; Xq 100000x220; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.istella.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 840514.6 | 840514.6..840514.6 | 1 | - | - | - | 2369.9 | - | finite=True, r2=-3.459e+24, rmse=1.554e+12 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 678615.5 | 678615.5..678615.5 | 1 | - | - | - | 2371.9 | - | finite=True, r2=-3.459e+24, rmse=1.554e+12 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 47324.9 | 47324.9..47324.9 | 1 | 17.760 | 14.339 | - | 1084.8 | - | finite=True, r2=-2.197e+24, rmse=1.238e+12 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "squared_error" | "squared_error" | "squared_error" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 21.5 | 21.5..21.5 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 20.4 | 20.4..20.4 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 4.1 | 4.1..4.1 | 1 | 5.171 | 4.907 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### sgd-reg / taxi (rows full, shape X 1000000x11; Xq 100000x11; y 1000000; yq 100000)

race: done, driver rc 0, log `logs/algos.sgd-reg.taxi.rows-full.log`, ran on ip-172-31-37-211.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 111774.0 | 111774.0..111774.0 | 1 | - | - | - | 604.9 | - | finite=True, r2=0.868127, rmse=5.783813 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 197241.6 | 197241.6..197241.6 | 1 | - | - | - | 634.8 | - | finite=True, r2=0.868127, rmse=5.783813 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5264.8 | 5264.8..5264.8 | 1 | 21.231 | 37.464 | - | 201.6 | - | finite=True, r2=0.880681, rmse=5.501638 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.005, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'constant', 'loss': 'squared_error', 'max_iter': 100, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows: None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

config: cuML benchmark (RAPIDS), MBSGDRegressor (https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 | 0.1 |
| eta0 | 0.005 | 0.005 | 0.005 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "constant" | "constant" | "constant" |
| loss | "squared_error" | "squared_error" | "squared_error" |
| max_iter | 100 | 100 | 100 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 8.8 | 8.8..8.8 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 11.9 | 11.9..11.9 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.6 | 0.6..0.6 | 1 | 15.334 | 20.762 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: predict(Xq)(Xq)

inference call, ours-fast: predict(Xq)(Xq)

inference call, sklearn-cpu: predict(Xq)(Xq)

### skewed-chi2 / istella (rows full, shape X 100000x220; Xq 1000x220; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.skewed-chi2.istella.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.5 | 7.5..7.5 | 1 | - | - | - | 1435.3 | - | kernel_rel_error=0.671898 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7.1 | 7.1..7.1 | 1 | - | - | - | 1436.9 | - | kernel_rel_error=0.671898 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.7 | 3.7..3.7 | 1 | 2.018 | 1.904 | - | 1253.8 | - | kernel_rel_error=0.671899 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_components': 256, 'random_state': 7, 'skewedness': 1.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 3.2 | 3.2..3.2 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 1.1 | 1.1..1.1 | 1 | 2.947 | 2.932 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

### skewed-chi2 / taxi (rows full, shape X 100000x11; Xq 1000x11; y 100000; yq 1000)

race: done, driver rc 0, log `logs/algos.skewed-chi2.taxi.rows-full.log`, ran on ip-172-31-43-215.ec2.internal

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.6 | 1.6..1.6 | 1 | - | - | - | 413.7 | - | kernel_rel_error=0.037749 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1.6 | 1.6..1.6 | 1 | - | - | - | 422.0 | - | kernel_rel_error=0.037749 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.6 | 0.6..0.6 | 1 | 2.896 | 2.866 | - | 241.2 | - | kernel_rel_error=0.037749 | - | LIKE-FOR-LIKE-SPAN | - | ok (measured this run) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_components': 256, 'random_state': 7, 'skewedness': 1.0}. Rows: None. Timed: None.

config: the board's own settings (no NVIDIA harness entry)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

Inference (each arm predicts with its own model from the fit rounds above):

| arm | batch | rows | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | quality | hash stable | comparability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| mojolearn FAST | Xq | - | 2.1 | 2.1..2.1 | 1 | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | ok |
| sklearn-cpu | Xq | - | 0.6 | 0.6..0.6 | 1 | 3.339 | 3.450 | - | - | - | LIKE-FOR-LIKE-SPAN | ok |

inference call, ours: transform(Xq)(Xq)

inference call, ours-fast: transform(Xq)(Xq)

inference call, sklearn-cpu: transform(Xq)(Xq)

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

