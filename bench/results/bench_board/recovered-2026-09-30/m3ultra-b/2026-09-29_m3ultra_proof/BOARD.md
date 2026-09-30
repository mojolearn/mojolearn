# mojolearn benchmark board

Generated 2026-09-29T14:14:09Z from `board.json` (schema `mojolearn-bench-board/1`).

> SMOKE RUN: `--rows 2000` is below the 1,000,000-row tree floor or the classical lane shapes; `--neural-shape small` is a plumbing shape. These numbers are plumbing checks, not results.

## Box

| field | value |
|---|---|
| vendor / API | apple / metal |
| GPU | Apple M3 Ultra |
| GPU driver | macOS 26.7 |
| CPU | Apple M3 Ultra (28 logical cores) |
| memory bytes | 274877906944 |
| OS | macOS 26.7 |
| Python | 3.13.15 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-macosx_11_0_arm64.whl, sha256 ee1187c950c29e791c906cb3beb1c45206e4d8534fea897bc839db4505b38548) |
| script commit | 7c264878053f0da621c3397b42c7309067382550 |
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

Races: 396 planned, 142 done, 77 failed, 177 pending. Cells: 721 (REFUSED 271, ok 450).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| algos | affinity-prop | istella | n_clusters | 184 | 184 | - | sklearn-cpu 184 |
| algos | affinity-prop | istella | silhouette (higher is better) | 0.090437 | 0.090437 | - | sklearn-cpu 0.090437 |
| algos | affinity-prop | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | affinity-prop | taxi | n_clusters | 148 | 148 | - | sklearn-cpu 148 |
| algos | affinity-prop | taxi | silhouette (higher is better) | 0.177010 | 0.177010 | - | sklearn-cpu 0.177010 |
| algos | affinity-prop | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| algos | bayesian-gmm | taxi | mean_log_likelihood (higher is better) | -3.296631 | -3.296631 | - | sklearn-cpu -1.672639 |
| algos | bisecting-kmeans | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | bisecting-kmeans | istella | silhouette (higher is better) | 0.127796 | 0.127796 | - | sklearn-cpu 0.121102 |
| algos | bisecting-kmeans | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.840882 |
| algos | bisecting-kmeans | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | bisecting-kmeans | taxi | silhouette (higher is better) | 0.153386 | 0.153386 | - | sklearn-cpu 0.146108 |
| algos | bisecting-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.540882 |
| algos | dict-learning | istella | component_sparsity | 0.104545 | 0.104545 | - | sklearn-cpu 0.104545 |
| algos | dict-learning | istella | relative_reconstruction_error (lower is better) | 0.529917 | 0.529917 | - | sklearn-cpu 0.529918 |
| algos | dict-learning | taxi | component_sparsity | 0.000000 | 0.000000 | - | sklearn-cpu 0.000000 |
| algos | dict-learning | taxi | relative_reconstruction_error (lower is better) | 0.471369 | 0.471369 | - | sklearn-cpu 0.467216 |
| algos | elliptic-envelope | istella | fraction_flagged | 0.100500 | 0.100500 | - | sklearn-cpu 0.096000 |
| algos | elliptic-envelope | istella | jaccard_vs_sklearn | 0.139130 | 0.139130 | - | sklearn-cpu 1.000000 |
| algos | elliptic-envelope | taxi | fraction_flagged | 0.101500 | 0.101500 | - | sklearn-cpu 0.091000 |
| algos | elliptic-envelope | taxi | jaccard_vs_sklearn | 0.750000 | 0.750000 | - | sklearn-cpu 1.000000 |
| algos | enet-cv | istella | r2 (higher is better) | 0.275639 | 0.275634 | - | sklearn-cpu 0.275634 |
| algos | enet-cv | istella | rmse (lower is better) | 0.675630 | 0.675633 | - | sklearn-cpu 0.675632 |
| algos | enet-cv | taxi | r2 (higher is better) | 0.930720 | 0.930720 | - | sklearn-cpu 0.930720 |
| algos | enet-cv | taxi | rmse (lower is better) | 3.844767 | 3.844766 | - | sklearn-cpu 3.844767 |
| algos | factor-analysis | istella | mean_log_likelihood (higher is better) | -2.5e+08 | -2.5e+08 | - | sklearn-cpu -2.5e+08 |
| algos | factor-analysis | taxi | mean_log_likelihood (higher is better) | -15.230434 | -15.230434 | - | sklearn-cpu -15.230489 |
| algos | fastica | istella | mean_abs_excess_kurtosis | 80.899842 | 80.895565 | - | sklearn-cpu 83.835013 |
| algos | fastica | taxi | mean_abs_excess_kurtosis | 4.956102 | 4.956103 | - | sklearn-cpu 4.948144 |
| algos | gaussian-rp | istella | mean_abs_distortion | 0.136908 | 0.136908 | - | sklearn-cpu 0.232702 |
| algos | gaussian-rp | taxi | mean_abs_distortion | 0.418718 | 0.418718 | - | sklearn-cpu 0.455943 |
| algos | kernel-pca | istella | subspace_cos_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | kernel-pca | taxi | subspace_cos_vs_sklearn | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| algos | lars | istella | r2 (higher is better) | -7.339e+30 | -1.609e+27 | - | sklearn-cpu -2.19e+49 |
| algos | lars | istella | rmse (lower is better) | 2.151e+15 | 3.185e+13 | - | sklearn-cpu 3.715e+24 |
| algos | lars | taxi | r2 (higher is better) | 0.930140 | 0.930140 | - | sklearn-cpu 0.930140 |
| algos | lars | taxi | rmse (lower is better) | 3.860829 | 3.860828 | - | sklearn-cpu 3.860823 |
| algos | lasso-cv | istella | r2 (higher is better) | 0.276319 | 0.276319 | - | sklearn-cpu 0.276321 |
| algos | lasso-cv | istella | rmse (lower is better) | 0.675313 | 0.675313 | - | sklearn-cpu 0.675312 |
| algos | lasso-cv | taxi | r2 (higher is better) | 0.930795 | 0.930795 | - | sklearn-cpu 0.930795 |
| algos | lasso-cv | taxi | rmse (lower is better) | 3.842696 | 3.842696 | - | sklearn-cpu 3.842697 |
| algos | lasso-lars | istella | r2 (higher is better) | 0.276169 | 0.263997 | - | sklearn-cpu 0.256262 |
| algos | lasso-lars | istella | rmse (lower is better) | 0.675383 | 0.681038 | - | sklearn-cpu 0.684607 |
| algos | lasso-lars | taxi | r2 (higher is better) | 0.930257 | 0.930257 | - | sklearn-cpu 0.930258 |
| algos | lasso-lars | taxi | rmse (lower is better) | 3.857584 | 3.857584 | - | sklearn-cpu 3.857579 |
| algos | lda | taxi-zones | perplexity | 54.130916 | 54.130923 | - | sklearn-cpu 54.602803 |
| algos | lda | text | perplexity | 267.071491 | 267.071498 | - | sklearn-cpu 266.677528 |
| algos | lle | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.743021 | - | - | sklearn-cpu 0.742862 |
| algos | lle | taxi | trustworthiness_k15 (higher is better, 1 at most) | - | - | - | sklearn-cpu 0.719153 |
| algos | logreg-cv | istella | accuracy (higher is better) | 0.922000 | 0.922000 | - | sklearn-cpu 0.922000 |
| algos | logreg-cv | istella | logloss (lower is better) | 0.192123 | 0.192201 | - | sklearn-cpu 0.192195 |
| algos | logreg-cv | taxi | accuracy (higher is better) | 0.763000 | 0.763000 | - | sklearn-cpu 0.763000 |
| algos | logreg-cv | taxi | logloss (lower is better) | 0.547045 | 0.547045 | - | sklearn-cpu 0.547041 |
| algos | louvain | istella | modularity | 0.813547 | 0.813547 | - | networkx-cpu 0.813989 |
| algos | louvain | istella | n_communities | 14 | 14 | - | networkx-cpu 14 |
| algos | louvain | taxi | modularity | 0.795766 | 0.795766 | - | networkx-cpu 0.800402 |
| algos | louvain | taxi | n_communities | 19 | 19 | - | networkx-cpu 17 |
| algos | mb-dict-learning | istella | component_sparsity | 0.104545 | 0.104545 | - | sklearn-cpu 0.104545 |
| algos | mb-dict-learning | istella | relative_reconstruction_error (lower is better) | 0.537705 | 0.537705 | - | sklearn-cpu 0.541634 |
| algos | mb-dict-learning | taxi | component_sparsity | 0.000000 | 0.000000 | - | sklearn-cpu 0.000000 |
| algos | mb-dict-learning | taxi | relative_reconstruction_error (lower is better) | 0.478903 | 0.478903 | - | sklearn-cpu 0.468998 |
| algos | mb-sparse-pca | istella | component_sparsity | 0.355114 | 0.355114 | - | sklearn-cpu 0.355114 |
| algos | mb-sparse-pca | istella | relative_reconstruction_error (lower is better) | 0.675542 | 0.675542 | - | sklearn-cpu 0.675542 |
| algos | mb-sparse-pca | taxi | component_sparsity | 0.250000 | 0.250000 | - | sklearn-cpu 0.250000 |
| algos | mb-sparse-pca | taxi | relative_reconstruction_error (lower is better) | 0.279922 | 0.279922 | - | sklearn-cpu 0.279922 |
| algos | min-cov-det | taxi | n_features | 11 | 11 | - | sklearn-cpu 11 |
| algos | min-cov-det | taxi | rel_diff_vs_sklearn | 0.133907 | 0.133906 | - | sklearn-cpu - |
| algos | minibatch-kmeans | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | minibatch-kmeans | istella | silhouette (higher is better) | 0.104964 | 0.104964 | - | sklearn-cpu 0.095856 |
| algos | minibatch-kmeans | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.529562 |
| algos | minibatch-kmeans | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| algos | minibatch-kmeans | taxi | silhouette (higher is better) | 0.166913 | 0.166913 | - | sklearn-cpu 0.143149 |
| algos | minibatch-kmeans | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.405726 |
| algos | nmf | istella | relative_reconstruction_error (lower is better) | 0.332851 | 0.332851 | - | sklearn-cpu 0.332851 |
| algos | nmf | taxi | relative_reconstruction_error (lower is better) | 0.089097 | 0.089097 | - | sklearn-cpu 0.089097 |
| algos | pa-clf | istella | accuracy (higher is better) | 0.908500 | 0.908500 | - | sklearn-cpu 0.907500 |
| algos | pa-clf | taxi | accuracy (higher is better) | 0.324500 | 0.324500 | - | sklearn-cpu 0.680000 |
| algos | pa-reg | istella | r2 (higher is better) | -0.783304 | -0.783304 | - | sklearn-cpu -0.524715 |
| algos | pa-reg | istella | rmse (lower is better) | 1.060094 | 1.060094 | - | sklearn-cpu 0.980225 |
| algos | pa-reg | taxi | r2 (higher is better) | 0.908105 | 0.908105 | - | sklearn-cpu 0.905999 |
| algos | pa-reg | taxi | rmse (lower is better) | 4.428049 | 4.428049 | - | sklearn-cpu 4.478495 |
| algos | perceptron | istella | accuracy (higher is better) | 0.883500 | 0.883500 | - | sklearn-cpu 0.914000 |
| algos | perceptron | taxi | accuracy (higher is better) | 0.404000 | 0.404000 | - | sklearn-cpu 0.590000 |
| algos | poly-count-sketch | istella | kernel_rel_error (lower is better) | 0.081609 | 0.081609 | - | sklearn-cpu 0.081609 |
| algos | poly-count-sketch | taxi | kernel_rel_error (lower is better) | 0.098761 | 0.098761 | - | sklearn-cpu 0.098761 |
| algos | randomized-svd | istella | relative_reconstruction_error (lower is better) | 0.0001122 | 0.0001122 | - | sklearn-cpu 0.0001122; torch-gpu 0.0001122 |
| algos | randomized-svd | taxi | relative_reconstruction_error (lower is better) | 0.027111 | 0.027111 | - | sklearn-cpu 0.027111; torch-gpu 0.027111 |
| algos | ridge-clf | istella | accuracy (higher is better) | 0.909000 | 0.909000 | - | sklearn-cpu 0.909000 |
| algos | ridge-clf | taxi | accuracy (higher is better) | 0.762500 | 0.762500 | - | sklearn-cpu 0.762500 |
| algos | sgd-clf | istella | accuracy (higher is better) | 0.916000 | 0.916000 | - | sklearn-cpu 0.917500 |
| algos | sgd-clf | taxi | accuracy (higher is better) | 0.637500 | 0.637500 | - | sklearn-cpu 0.731000 |
| algos | sgd-ocsvm | istella | fraction_flagged | 0.000000 | 0.000000 | - | sklearn-cpu 0.013500 |
| algos | sgd-ocsvm | istella | jaccard_vs_sklearn | 0.000000 | 0.000000 | - | sklearn-cpu 1.000000 |
| algos | sgd-ocsvm | taxi | fraction_flagged | 0.110000 | 0.110000 | - | sklearn-cpu 0.025000 |
| algos | sgd-ocsvm | taxi | jaccard_vs_sklearn | 0.189427 | 0.189427 | - | sklearn-cpu 1.000000 |
| algos | sgd-reg | istella | r2 (higher is better) | -4.301e+07 | -4.301e+07 | - | sklearn-cpu -7.938e+07 |
| algos | sgd-reg | istella | rmse (lower is better) | 5206.367298 | 5206.381796 | - | sklearn-cpu 7072.558134 |
| algos | sgd-reg | taxi | r2 (higher is better) | 0.930328 | 0.930328 | - | sklearn-cpu 0.930762 |
| algos | sgd-reg | taxi | rmse (lower is better) | 3.855633 | 3.855633 | - | sklearn-cpu 3.843604 |
| algos | skewed-chi2 | istella | kernel_rel_error (lower is better) | 0.740524 | 0.740524 | - | sklearn-cpu 0.740524 |
| algos | skewed-chi2 | taxi | kernel_rel_error (lower is better) | 0.037983 | 0.037983 | - | sklearn-cpu 0.037983 |
| algos | sparse-pca | istella | component_sparsity | 0.509659 | 0.509659 | - | sklearn-cpu 0.509659 |
| algos | sparse-pca | istella | relative_reconstruction_error (lower is better) | 0.675875 | 0.675875 | - | sklearn-cpu 0.675875 |
| algos | sparse-pca | taxi | component_sparsity | 0.693182 | 0.693182 | - | sklearn-cpu 0.693182 |
| algos | sparse-pca | taxi | relative_reconstruction_error (lower is better) | 0.282172 | 0.282172 | - | sklearn-cpu 0.282172 |
| algos | sparse-rp | istella | mean_abs_distortion | 0.077581 | 0.077581 | - | sklearn-cpu 0.288047 |
| algos | sparse-rp | taxi | mean_abs_distortion | 0.287559 | 0.287559 | - | sklearn-cpu 0.460698 |
| classical | dbscan | istella | n_clusters | 18 | 18 | - | sklearn-cpu 18 |
| classical | dbscan | istella | noise_fraction | 0.611000 | 0.611000 | - | sklearn-cpu 0.611000 |
| classical | dbscan | istella | rows | 2000 | 2000 | - | sklearn-cpu 2000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | taxi | n_clusters | 11 | 11 | - | sklearn-cpu 11 |
| classical | dbscan | taxi | noise_fraction | 0.366000 | 0.366000 | - | sklearn-cpu 0.366000 |
| classical | dbscan | taxi | rows | 2000 | 2000 | - | sklearn-cpu 2000 |
| classical | dbscan | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | taxi | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | hdbscan | istella | n_clusters | 5 | 5 | - | sklearn-cpu 5 |
| classical | hdbscan | istella | noise_fraction | 0.492500 | 0.492500 | - | sklearn-cpu 0.468000 |
| classical | hdbscan | istella | rows | 2000 | 2000 | - | sklearn-cpu 2000 |
| classical | hdbscan | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.931372 |
| classical | hdbscan | istella | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu 0.972500 |
| classical | hdbscan | taxi | n_clusters | 4 | 4 | - | sklearn-cpu 4 |
| classical | hdbscan | taxi | noise_fraction | 0.211500 | 0.211500 | - | sklearn-cpu 0.206000 |
| classical | hdbscan | taxi | rows | 2000 | 2000 | - | sklearn-cpu 2000 |
| classical | hdbscan | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.991083 |
| classical | hdbscan | taxi | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu 0.994500 |
| classical | kde | istella | mean_log_likelihood (higher is better) | -404.680084 | -404.680084 | - | sklearn-cpu -402.006921 |
| classical | kde | istella | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | -11.508770 | -11.508770 | - | sklearn-cpu -11.508770 |
| classical | kde | taxi | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kmeans | istella | inertia (lower is better) | 4.423e+14 | 4.423e+14 | - | sklearn-cpu 4.423e+14; torch-gpu 4.423e+14 |
| classical | kmeans | istella | inertia_over_ours | 0.999996 | 1.000000 | - | sklearn-cpu 1.000002; torch-gpu 1.000005 |
| classical | kmeans | istella | n_iter | 21 | 21 | - | sklearn-cpu 20; torch-gpu 20 |
| classical | kmeans | taxi | inertia (lower is better) | 6524.265344 | 6524.265344 | - | sklearn-cpu 6524.265571; torch-gpu 6524.265588 |
| classical | kmeans | taxi | inertia_over_ours | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu 1.000000 |
| classical | kmeans | taxi | n_iter | 21 | 21 | - | sklearn-cpu 20; torch-gpu 20 |
| classical | knn | istella | recall_at_10 (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu 1.000000 |
| classical | knn | istella | rows_with_repeated_ids | 0 | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | knn | taxi | recall_at_10 (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu 1.000000 |
| classical | knn | taxi | rows_with_repeated_ids | 0 | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | -24.390212 | -24.129541 | - | sklearn-cpu -0.027117; torch-gpu - |
| classical | ols | istella | rmse (lower is better) | 4.771147 | 4.746592 | - | sklearn-cpu 0.959621; torch-gpu - |
| classical | ols | taxi | r2 (higher is better) | 0.889542 | 0.889788 | - | sklearn-cpu 0.907724; torch-gpu - |
| classical | ols | taxi | rmse (lower is better) | 6.587113 | 6.579774 | - | sklearn-cpu 6.020627; torch-gpu - |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu - |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | 0.999988 | 0.999988 | - | sklearn-cpu 0.999987; torch-gpu - |
| classical | svc | istella | accuracy (higher is better) | 0.855469 | 0.855469 | - | sklearn-cpu 0.855469 |
| classical | svc | istella | n_support | 94 | 94 | - | sklearn-cpu 94 |
| classical | svc | taxi | accuracy (higher is better) | 0.765625 | 0.765625 | - | sklearn-cpu 0.765625 |
| classical | svc | taxi | n_support | 151 | 151 | - | sklearn-cpu 151 |
| classical2 | agglomerative | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | agglomerative | istella | silhouette (higher is better) | 0.652303 | 0.652303 | - | sklearn-cpu 0.652303 |
| classical2 | agglomerative | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical2 | agglomerative | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | agglomerative | taxi | silhouette (higher is better) | 0.522237 | 0.522237 | - | sklearn-cpu 0.522237 |
| classical2 | agglomerative | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical2 | arima | synthetic | forecast_rmse (lower is better) | 1.515540 | 1.515518 | - | statsmodels-cpu 1.515423 |
| classical2 | arima | synthetic | insample_rmse (lower is better) | 0.999341 | 0.999342 | - | statsmodels-cpu 0.999338 |
| classical2 | arima | synthetic | mean_aic (lower is better) | 5680.971687 | 5680.976967 | - | statsmodels-cpu 5680.957160 |
| classical2 | arima | synthetic | mean_llf (higher is better) | -2836.485844 | -2836.488483 | - | statsmodels-cpu -2836.478580 |
| classical2 | elasticnet | istella | r2 (higher is better) | 0.275640 | 0.275634 | - | sklearn-cpu 0.275634 |
| classical2 | elasticnet | istella | rmse (lower is better) | 0.675630 | 0.675632 | - | sklearn-cpu 0.675632 |
| classical2 | elasticnet | taxi | r2 (higher is better) | 0.930720 | 0.930720 | - | sklearn-cpu 0.930720 |
| classical2 | elasticnet | taxi | rmse (lower is better) | 3.844768 | 3.844767 | - | sklearn-cpu 3.844767 |
| classical2 | ets | synthetic | forecast_rmse (lower is better) | 0.984473 | 0.984392 | - | statsmodels-cpu 0.984418 |
| classical2 | ets | synthetic | insample_rmse (lower is better) | 0.990971 | 0.990971 | - | statsmodels-cpu 0.991812 |
| classical2 | gmm | taxi | bic (lower is better) | -65102.837429 | -68064.910372 | - | sklearn-cpu -69061.429868 |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | 12.478869 | 12.918593 | - | sklearn-cpu 13.245844 |
| classical2 | gmm | taxi | n_iter | 11 | 25 | - | sklearn-cpu 100 |
| classical2 | gpc | istella | accuracy (higher is better) | 0.919000 | 0.919000 | - | sklearn-cpu 0.919000 |
| classical2 | gpc | istella | logloss (lower is better) | 0.215260 | 0.215232 | - | sklearn-cpu 0.215253 |
| classical2 | gpc | istella | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | gpc | taxi | accuracy (higher is better) | 0.764000 | 0.764000 | - | sklearn-cpu 0.764000 |
| classical2 | gpc | taxi | logloss (lower is better) | 0.541865 | 0.541835 | - | sklearn-cpu 0.541862 |
| classical2 | gpc | taxi | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | gpr | istella | mean_log_predictive_density (higher is better) | -5.391598 | -5.391488 | - | sklearn-cpu -5.391831 |
| classical2 | gpr | istella | r2 (higher is better) | 0.228699 | 0.228689 | - | sklearn-cpu 0.228693 |
| classical2 | gpr | istella | rmse (lower is better) | 0.697178 | 0.697182 | - | sklearn-cpu 0.697180 |
| classical2 | gpr | taxi | mean_log_predictive_density (higher is better) | -279.853051 | -279.852141 | - | sklearn-cpu -279.891365 |
| classical2 | gpr | taxi | r2 (higher is better) | 0.838374 | 0.838375 | - | sklearn-cpu 0.838375 |
| classical2 | gpr | taxi | rmse (lower is better) | 5.872494 | 5.872471 | - | sklearn-cpu 5.872469 |
| classical2 | ivf | istella | recall_at_10 (higher is better) | 1.000000 | 0.998437 | - | faiss-cpu 1.000000 |
| classical2 | ivf | istella | rows_with_repeated_ids | 0 | 0 | - | faiss-cpu 0 |
| classical2 | ivf | taxi | recall_at_10 (higher is better) | 0.956250 | 0.956250 | - | faiss-cpu 0.903125 |
| classical2 | ivf | taxi | rows_with_repeated_ids | 0 | 0 | - | faiss-cpu 0 |
| classical2 | kernel-ridge | istella | r2 (higher is better) | 0.299806 | 0.299806 | - | sklearn-cpu 0.299806 |
| classical2 | kernel-ridge | istella | rmse (lower is better) | 0.664264 | 0.664264 | - | sklearn-cpu 0.664264 |
| classical2 | kernel-ridge | taxi | r2 (higher is better) | 0.650912 | 0.650912 | - | sklearn-cpu 0.650912 |
| classical2 | kernel-ridge | taxi | rmse (lower is better) | 8.630457 | 8.630456 | - | sklearn-cpu 8.630457 |
| classical2 | knn-clf | istella | accuracy (higher is better) | 0.911000 | 0.911000 | - | sklearn-cpu 0.911000 |
| classical2 | knn-clf | taxi | accuracy (higher is better) | 0.727500 | 0.727500 | - | sklearn-cpu 0.727500 |
| classical2 | knn-reg | istella | r2 (higher is better) | 0.239429 | 0.239429 | - | sklearn-cpu 0.239429 |
| classical2 | knn-reg | istella | rmse (lower is better) | 0.692311 | 0.692311 | - | sklearn-cpu 0.692311 |
| classical2 | knn-reg | taxi | r2 (higher is better) | 0.860399 | 0.860399 | - | sklearn-cpu 0.860399 |
| classical2 | knn-reg | taxi | rmse (lower is better) | 5.457721 | 5.457721 | - | sklearn-cpu 5.457721 |
| classical2 | lasso | istella | r2 (higher is better) | 0.276322 | 0.276321 | - | sklearn-cpu 0.276321 |
| classical2 | lasso | istella | rmse (lower is better) | 0.675312 | 0.675312 | - | sklearn-cpu 0.675312 |
| classical2 | lasso | taxi | r2 (higher is better) | 0.930249 | 0.930249 | - | sklearn-cpu 0.930249 |
| classical2 | lasso | taxi | rmse (lower is better) | 3.857818 | 3.857817 | - | sklearn-cpu 3.857817 |
| classical2 | linearsvc | istella | accuracy (higher is better) | 0.912500 | 0.913500 | - | sklearn-cpu 0.913500 |
| classical2 | linearsvc | taxi | accuracy (higher is better) | 0.763000 | 0.763000 | - | sklearn-cpu 0.762500 |
| classical2 | linearsvr | istella | r2 (higher is better) | -0.092549 | -0.092549 | - | sklearn-cpu -0.087102 |
| classical2 | linearsvr | istella | rmse (lower is better) | 0.829759 | 0.829759 | - | sklearn-cpu 0.827688 |
| classical2 | linearsvr | taxi | r2 (higher is better) | -0.420609 | 0.461132 | - | sklearn-cpu 0.923553 |
| classical2 | linearsvr | taxi | rmse (lower is better) | 17.410203 | 10.722794 | - | sklearn-cpu 4.038760 |
| classical2 | logreg | istella | accuracy (higher is better) | 0.923500 | 0.923500 | - | sklearn-cpu 0.923500 |
| classical2 | logreg | istella | logloss (lower is better) | 0.204812 | 0.204798 | - | sklearn-cpu 0.204942 |
| classical2 | logreg | istella | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | 0.763500 | 0.763500 | - | sklearn-cpu 0.763500 |
| classical2 | logreg | taxi | logloss (lower is better) | 0.547874 | 0.547874 | - | sklearn-cpu 0.547867 |
| classical2 | logreg | taxi | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | nystroem | istella | kernel_rel_error (lower is better) | 0.028603 | 0.028603 | - | sklearn-cpu 0.028046 |
| classical2 | nystroem | taxi | kernel_rel_error (lower is better) | 0.017836 | 0.017836 | - | sklearn-cpu 0.021057 |
| classical2 | rbf-sampler | istella | kernel_rel_error (lower is better) | 0.135214 | 0.135214 | - | sklearn-cpu 0.139828 |
| classical2 | rbf-sampler | taxi | kernel_rel_error (lower is better) | 0.134474 | 0.134474 | - | sklearn-cpu 0.130369 |
| classical2 | ridge | istella | r2 (higher is better) | 0.015703 | 0.015427 | - | sklearn-cpu 0.016246 |
| classical2 | ridge | istella | rmse (lower is better) | 0.787580 | 0.787690 | - | sklearn-cpu 0.787363 |
| classical2 | ridge | taxi | r2 (higher is better) | 0.930198 | 0.930198 | - | sklearn-cpu 0.930198 |
| classical2 | ridge | taxi | rmse (lower is better) | 3.859238 | 3.859240 | - | sklearn-cpu 3.859233 |
| classical2 | spectral-embedding | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.858681 | 0.858681 | - | sklearn-cpu 0.858691 |
| classical2 | spectral-embedding | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.708797 | 0.690399 | - | sklearn-cpu 0.687460 |
| classical2 | spectral | istella | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | spectral | istella | silhouette (higher is better) | 0.058626 | 0.058626 | - | sklearn-cpu 0.058626 |
| classical2 | spectral | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical2 | spectral | taxi | n_clusters | 8 | 8 | - | sklearn-cpu 8 |
| classical2 | spectral | taxi | silhouette (higher is better) | 0.084290 | 0.084290 | - | sklearn-cpu 0.083875 |
| classical2 | spectral | taxi | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 0.993023 |
| classical2 | svr | istella | r2 (higher is better) | 0.235309 | 0.235325 | - | sklearn-cpu 0.235295 |
| classical2 | svr | istella | rmse (lower is better) | 0.694184 | 0.694177 | - | sklearn-cpu 0.694190 |
| classical2 | svr | taxi | r2 (higher is better) | 0.654429 | 0.654427 | - | sklearn-cpu 0.654426 |
| classical2 | svr | taxi | rmse (lower is better) | 8.586880 | 8.586900 | - | sklearn-cpu 8.586920 |
| classical2 | tsvd | istella | explained_variance_ratio_sum (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000 |
| classical2 | tsvd | istella | relative_reconstruction_error (lower is better) | 0.0001106 | 0.0001106 | - | sklearn-cpu 0.0001083 |
| classical2 | tsvd | taxi | explained_variance_ratio_sum (higher is better) | 0.997435 | 0.997435 | - | sklearn-cpu 0.997434 |
| classical2 | tsvd | taxi | relative_reconstruction_error (lower is better) | 0.026699 | 0.026699 | - | sklearn-cpu 0.026699 |
| classical2 | umap | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.949116 | 0.943151 | - | umap-learn-cpu 0.945949; umap-learn-cpu-unseeded 0.944117 |
| classical2 | umap | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.973998 | 0.969395 | - | umap-learn-cpu 0.973838; umap-learn-cpu-unseeded 0.973131 |
| trees | et | istella | logloss (lower is better) | 0.532243 | 0.532243 | - | sklearn-et-cpu 0.542278; lightgbm-cpu 0.273911 |
| trees | et | istella | auc (higher is better) | 0.857018 | 0.857018 | - | sklearn-et-cpu 0.858858; lightgbm-cpu 0.877419 |
| trees | et | taxi | logloss (lower is better) | 0.644383 | 0.644383 | - | sklearn-et-cpu 0.652006; lightgbm-cpu 0.864346 |
| trees | et | taxi | auc (higher is better) | 0.573465 | 0.573465 | - | sklearn-et-cpu 0.566032; lightgbm-cpu 0.565876 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | 0.596223 | 0.570840 | - | catboost-cpu 0.612688; xgboost-cpu 0.869732; lightgbm-cpu 0.622127 |
| trees | gbdt-categorical | taxi | auc (higher is better) | 0.547703 | 0.564749 | - | catboost-cpu 0.534313; xgboost-cpu 0.501489; lightgbm-cpu 0.564264 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | 0.360172 | 0.347768 | - | catboost-cpu 0.354825; xgboost-cpu 0.437510 |
| trees | gbdt-depthwise | istella | auc (higher is better) | 0.892237 | 0.892015 | - | catboost-cpu 0.890673; xgboost-cpu 0.893986 |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | 0.607221 | 0.720729 | - | catboost-cpu 0.616489; xgboost-cpu 0.626239 |
| trees | gbdt-depthwise | taxi | auc (higher is better) | 0.566061 | 0.559244 | - | catboost-cpu 0.559791; xgboost-cpu 0.549320 |
| trees | gbdt-lossguide | istella | logloss (lower is better) | 0.430454 | 0.430332 | - | catboost-cpu 0.354825; xgboost-cpu 0.437510; lightgbm-cpu 0.403109 |
| trees | gbdt-lossguide | istella | auc (higher is better) | 0.894370 | 0.894619 | - | catboost-cpu 0.890673; xgboost-cpu 0.893986; lightgbm-cpu 0.897661 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | 0.649291 | 0.646470 | - | catboost-cpu 0.616489; xgboost-cpu 0.626239; lightgbm-cpu 0.593295 |
| trees | gbdt-lossguide | taxi | auc (higher is better) | 0.554204 | 0.552868 | - | catboost-cpu 0.559791; xgboost-cpu 0.549320; lightgbm-cpu 0.565812 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | 0.482633 | 0.482695 | - | catboost-cpu 0.459695; xgboost-cpu 0.687564; lightgbm-cpu 0.697130 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | 0.886530 | 0.886532 | - | catboost-cpu 0.886352; xgboost-cpu 0.887958; lightgbm-cpu 0.889364 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | 1.259573 | 1.259573 | - | catboost-cpu 1.258732; xgboost-cpu 1.459059; lightgbm-cpu 1.408965 |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | 0.460396 | 0.460396 | - | catboost-cpu 0.412360; xgboost-cpu 0.431244; lightgbm-cpu 0.449182 |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | 0.615095 | 0.615034 | - | catboost-cpu 0.624304; xgboost-cpu 0.586953 |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | 0.553705 | 0.553685 | - | catboost-cpu 0.561860; xgboost-cpu 0.530981 |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | 0.730104 | 0.730103 | - | catboost-cpu 0.730439; xgboost-cpu 0.698987 |
| trees | gbdt-rank-yetirank | istella | ndcg10 (higher is better) | 0.608340 | 0.608837 | - | catboost-cpu 0.601311; xgboost-cpu 0.579654; lightgbm-cpu 0.601041 |
| trees | gbdt-rank-yetirank | istella | ndcg5 (higher is better) | 0.546274 | 0.546157 | - | catboost-cpu 0.539947; xgboost-cpu 0.519897; lightgbm-cpu 0.540794 |
| trees | gbdt-rank-yetirank | istella | map (higher is better) | 0.718820 | 0.721309 | - | catboost-cpu 0.713065; xgboost-cpu 0.690516; lightgbm-cpu 0.715387 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | 0.297860 | 0.297848 | - | catboost-cpu 0.285860 |
| trees | gbdt-symmetric | istella | auc (higher is better) | 0.911488 | 0.911564 | - | catboost-cpu 0.916973 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | 0.603281 | 0.599359 | - | catboost-cpu 0.581298 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | 0.562043 | 0.563391 | - | catboost-cpu 0.576648 |
| trees | iforest | istella | auc (higher is better) | 0.763032 | 0.763032 | - | sklearn-iforest-cpu 0.792792 |
| trees | iforest | taxi | auc (higher is better) | 0.542833 | 0.542833 | - | sklearn-iforest-cpu 0.538780 |
| trees | rf | istella | logloss (lower is better) | 0.444021 | 0.444021 | - | sklearn-rf-cpu 0.486884; lightgbm-cpu 0.262548 |
| trees | rf | istella | auc (higher is better) | 0.876094 | 0.876094 | - | sklearn-rf-cpu 0.870452; lightgbm-cpu 0.877515 |
| trees | rf | taxi | logloss (lower is better) | 0.669411 | 0.669411 | - | sklearn-rf-cpu 0.607659; lightgbm-cpu 0.772297 |
| trees | rf | taxi | auc (higher is better) | 0.578982 | 0.578982 | - | sklearn-rf-cpu 0.571638; lightgbm-cpu 0.578311 |

## Trees

### et / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/et.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 97.7 | 97.7..97.7 | 1 | - | - | - | 3821.2 | - | auc=0.857018, logloss=0.532243 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 87.0 | 87.0..87.0 | 1 | - | - | - | 4242.2 | - | auc=0.857018, logloss=0.532243 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 41.9 | 41.9..41.9 | 1 | 2.334 | 2.079 | - | 3719.7 | - | auc=0.858858, logloss=0.542278 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 8487.3 | 8487.3..8487.3 | 1 | 0.012 | 0.010 | - | 14066.2 | - | auc=0.877419, logloss=0.273911 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:9042,ours:15839,ours-ab:15839,sklearn-et-cpu:15865 spread=0.4301 verdict=NOT-COMPARABLE`

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

### et / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/et.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 108.4 | 108.4..108.4 | 1 | - | - | - | 1139.9 | - | auc=0.573465, logloss=0.644383 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 102.2 | 102.2..102.2 | 1 | - | - | - | 1172.5 | - | auc=0.573465, logloss=0.644383 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 122.3 | 122.3..122.3 | 1 | 0.886 | 0.836 | - | 1106.5 | - | auc=0.566032, logloss=0.652006 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 1736.3 | 1736.3..1736.3 | 1 | 0.062 | 0.059 | - | 1870.8 | - | auc=0.565876, logloss=0.864346 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:2045,ours:41616,ours-ab:41616,sklearn-et-cpu:40790 spread=0.9509 verdict=NOT-COMPARABLE`

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

### gbdt-categorical / taxi (rows 2000, shape taxicat-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-categorical.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3417.1 | 3417.1..3417.1 | 1 | - | - | - | 952.1 | - | auc=0.564749, logloss=0.570840 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3515.1 | 3515.1..3515.1 | 1 | - | - | - | 1219.0 | - | auc=0.547703, logloss=0.596223 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 213.7 | 213.7..213.7 | 1 | 15.991 | 16.450 | - | 1363.0 | - | auc=0.534313, logloss=0.612688 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 2464.4 | 2464.4..2464.4 | 1 | 1.387 | 1.426 | - | 1246.4 | - | auc=0.501489, logloss=0.869732 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 1383.4 | 1383.4..1383.4 | 1 | 2.470 | 2.541 | - | 1240.7 | - | auc=0.564264, logloss=0.622127 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-categorical arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:3365,lightgbm-cpu:2804,ours:4138,ours-ab:4019,xgboost-cpu:4523 spread=0.3801 verdict=NOT-COMPARABLE`

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
| max_depth | 6 | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 | 64 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 0.0 | - | 0.0 | 0.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
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

### gbdt-depthwise / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1561.5 | 1561.5..1561.5 | 1 | - | - | - | 2709.2 | - | auc=0.892015, logloss=0.347768 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1567.5 | 1567.5..1567.5 | 1 | - | - | - | 3631.1 | - | auc=0.892237, logloss=0.360172 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 383.1 | 383.1..383.1 | 1 | 4.076 | 4.092 | - | 4530.8 | - | auc=0.890673, logloss=0.354825 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 510.4 | 510.4..510.4 | 1 | 3.060 | 3.071 | - | 5093.7 | - | auc=0.893986, logloss=0.437510 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:3373,ours:3672,ours-ab:3479,xgboost-cpu:3207 spread=0.1266 verdict=NOT-COMPARABLE`

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
| max_depth | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 |
| min_child_weight | - | null | null | 0.0 |
| min_samples_leaf | 1 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | "Min" | "Min" | - |
| random_strength | 0.0 | 0.0 | 0.0 | - |
| reg_alpha | - | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### gbdt-depthwise / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-depthwise.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 836.8 | 836.8..836.8 | 1 | - | - | - | 874.5 | - | auc=0.559244, logloss=0.720729 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 877.9 | 877.9..877.9 | 1 | - | - | - | 958.4 | - | auc=0.566061, logloss=0.607221 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 119.6 | 119.6..119.6 | 1 | 6.997 | 7.341 | - | 1029.3 | - | auc=0.559791, logloss=0.616489 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 525.1 | 525.1..525.1 | 1 | 1.593 | 1.672 | - | 1086.8 | - | auc=0.549320, logloss=0.626239 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:2660,ours:2621,ours-ab:2707,xgboost-cpu:3678 spread=0.2874 verdict=NOT-COMPARABLE`

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
| max_depth | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 |
| min_child_weight | - | null | null | 0.0 |
| min_samples_leaf | 1 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | "Min" | "Min" | - |
| random_strength | 0.0 | 0.0 | 0.0 | - |
| reg_alpha | - | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | "Cosine" | "Cosine" | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | null | null | null | 1.0 |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: xgboost-cpu boosting_type: different vocabularies: ours and CatBoost 'Plain' (not Ordered), XGBoost booster 'gbtree'; both plain gradient boosting

accepted difference: xgboost-cpu min_child_weight: ours takes min_child_hessian only with a Newton score and Depthwise runs Cosine (CatBoost CPU has no Newton score), so ours has no hessian floor: XGBoost min_child_weight 0

accepted difference: xgboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### gbdt-lossguide / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3497.5 | 3497.5..3497.5 | 1 | - | - | - | 2718.8 | - | auc=0.894619, logloss=0.430332 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3476.0 | 3476.0..3476.0 | 1 | - | - | - | 3636.7 | - | auc=0.894370, logloss=0.430454 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 536.0 | 536.0..536.0 | 1 | 6.525 | 6.485 | - | 4532.8 | - | auc=0.890673, logloss=0.354825 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 1895.1 | 1895.1..1895.1 | 1 | 1.846 | 1.834 | - | 5104.8 | - | auc=0.893986, logloss=0.437510 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 1198.2 | 1198.2..1198.2 | 1 | 2.919 | 2.901 | - | 4835.7 | - | auc=0.897661, logloss=0.403109 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:3373,lightgbm-cpu:1866,ours:4164,ours-ab:3639,xgboost-cpu:3207 spread=0.5519 verdict=NOT-COMPARABLE`

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
| max_depth | 6 | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 | 64 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 0.0 | - | 0.0 | 0.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
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

### gbdt-lossguide / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-lossguide.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2901.4 | 2901.4..2901.4 | 1 | - | - | - | 885.3 | - | auc=0.552868, logloss=0.646470 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3098.3 | 3098.3..3098.3 | 1 | - | - | - | 970.5 | - | auc=0.554204, logloss=0.649291 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 125.9 | 125.9..125.9 | 1 | 23.053 | 24.617 | - | 1023.8 | - | auc=0.559791, logloss=0.616489 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 2090.1 | 2090.1..2090.1 | 1 | 1.388 | 1.482 | - | 1097.0 | - | auc=0.549320, logloss=0.626239 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 995.9 | 995.9..995.9 | 1 | 2.913 | 3.111 | - | 838.1 | - | auc=0.565812, logloss=0.593295 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:2660,lightgbm-cpu:1962,ours:4246,ours-ab:3982,xgboost-cpu:3678 spread=0.5379 verdict=NOT-COMPARABLE`

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
| max_depth | 6 | 6 | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 | 64 | 64 |
| min_child_weight | - | 0.001 | 0.0 | 0.0 | 0.0 |
| min_samples_leaf | 1 | 20 | 1 | 1 | - |
| min_split_gain | - | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 | 100 |
| nan_mode | "Min" | - | "Min" | "Min" | - |
| random_strength | 0.0 | - | 0.0 | 0.0 | - |
| reg_alpha | - | 0.0 | - | - | 0.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
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

### gbdt-multiclass / istella (rows 2000, shape istellamc-2000x220)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1327.0 | 1327.0..1327.0 | 1 | - | - | - | 2806.8 | - | accuracy=0.886532, mlogloss=0.482695 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1400.3 | 1400.3..1400.3 | 1 | - | - | - | 3778.0 | - | accuracy=0.886530, mlogloss=0.482633 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 747.8 | 747.8..747.8 | 1 | 1.775 | 1.873 | - | 4728.6 | - | accuracy=0.886352, mlogloss=0.459695 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 2539.7 | 2539.7..2539.7 | 1 | 0.522 | 0.551 | - | 5350.9 | - | accuracy=0.887958, mlogloss=0.687564 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 5758.5 | 5758.5..5758.5 | 1 | 0.230 | 0.243 | - | 5066.8 | - | accuracy=0.889364, mlogloss=0.697130 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6400,lightgbm-cpu:1794,ours:6400,ours-ab:6400,xgboost-cpu:2526 spread=0.7197 verdict=NOT-COMPARABLE`

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

### gbdt-multiclass / taxi (rows 2000, shape taximc-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 719.5 | 719.5..719.5 | 1 | - | - | - | 910.4 | - | accuracy=0.460396, mlogloss=1.259573 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 713.7 | 713.7..713.7 | 1 | - | - | - | 994.6 | - | accuracy=0.460396, mlogloss=1.259573 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 118.9 | 118.9..118.9 | 1 | 6.052 | 6.004 | - | 1070.5 | - | accuracy=0.412360, mlogloss=1.258732 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 2030.0 | 2030.0..2030.0 | 1 | 0.354 | 0.352 | - | 1181.1 | - | accuracy=0.431244, mlogloss=1.459059 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 3923.6 | 3923.6..3923.6 | 1 | 0.183 | 0.182 | - | 901.8 | - | accuracy=0.449182, mlogloss=1.408965 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6252,lightgbm-cpu:1914,ours:6400,ours-ab:6400,xgboost-cpu:3478 spread=0.7009 verdict=NOT-COMPARABLE`

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

### gbdt-rank-pairlogit / istella (rows 2000, shape istellarank-1928x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-pairlogit.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1253.4 | 1253.4..1253.4 | 1 | - | - | - | 2889.4 | - | map=0.730103, ndcg10=0.615034, ndcg5=0.553685 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1163.1 | 1163.1..1163.1 | 1 | - | - | - | 4130.2 | - | map=0.730104, ndcg10=0.615095, ndcg5=0.553705 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1552.4 | 1552.4..1552.4 | 1 | 0.807 | 0.749 | - | 5362.8 | - | map=0.730439, ndcg10=0.624304, ndcg5=0.561860 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 520.6 | 520.6..520.6 | 1 | 2.407 | 2.234 | - | 5848.9 | - | map=0.698987, ndcg10=0.586953, ndcg5=0.530981 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-pairlogit arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6368,ours:6400,ours-ab:6400,xgboost-cpu:2868 spread=0.5519 verdict=NOT-COMPARABLE`

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

### gbdt-rank-yetirank / istella (rows 2000, shape istellarank-1928x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-yetirank.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7551.5 | 7551.5..7551.5 | 1 | - | - | - | 2925.9 | - | map=0.721309, ndcg10=0.608837, ndcg5=0.546157 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8897.4 | 8897.4..8897.4 | 1 | - | - | - | 4168.1 | - | map=0.718820, ndcg10=0.608340, ndcg5=0.546274 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1294.8 | 1294.8..1294.8 | 1 | 5.832 | 6.871 | - | 5397.3 | - | map=0.713065, ndcg10=0.601311, ndcg5=0.539947 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 522.1 | 522.1..522.1 | 1 | 14.463 | 17.041 | - | 5811.6 | - | map=0.690516, ndcg10=0.579654, ndcg5=0.519897 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 1218.3 | 1218.3..1218.3 | 1 | 6.198 | 7.303 | - | 5817.7 | - | map=0.715387, ndcg10=0.601041, ndcg5=0.540794 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-yetirank arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6400,lightgbm-cpu:1890,ours:6400,ours-ab:6400,xgboost-cpu:3348 spread=0.7047 verdict=NOT-COMPARABLE`

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

### gbdt-symmetric / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1109.9 | 1109.9..1109.9 | 1 | - | - | - | 2599.3 | - | auc=0.911564, logloss=0.297848 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1068.8 | 1068.8..1068.8 | 1 | - | - | - | 3529.2 | - | auc=0.911488, logloss=0.297860 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1179.6 | 1179.6..1179.6 | 1 | 0.941 | 0.906 | - | 4437.4 | - | auc=0.916973, logloss=0.285860 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:6400,ours:6400,ours-ab:6400 spread=0.0000 verdict=COMPARABLE`

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
| max_depth | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 |
| min_child_weight | - | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | - | null | null |
| n_estimators | 100 | 100 | 100 |
| nan_mode | "Min" | "Min" | "Min" |
| random_strength | 1.0 | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### gbdt-symmetric / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 254.2 | 254.2..254.2 | 1 | - | - | - | 818.4 | - | auc=0.563391, logloss=0.599359 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 528.6 | 528.6..528.6 | 1 | - | - | - | 902.1 | - | auc=0.562043, logloss=0.603281 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 57.3 | 57.3..57.3 | 1 | 4.437 | 9.227 | - | 968.4 | - | auc=0.576648, logloss=0.581298 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:6060,ours:6400,ours-ab:6400 spread=0.0531 verdict=COMPARABLE`

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
| max_depth | 6 | 6 | 6 |
| max_leaves | 64 | 64 | 64 |
| min_child_weight | - | null | null |
| min_samples_leaf | 1 | 1 | 1 |
| min_split_gain | - | null | null |
| n_estimators | 100 | 100 | 100 |
| nan_mode | "Min" | "Min" | "Min" |
| random_strength | 1.0 | 1.0 | 1.0 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### iforest / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/iforest.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 63.9 | 63.9..63.9 | 1 | - | - | - | 2589.0 | - | auc=0.763032 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 63.0 | 63.0..63.0 | 1 | - | - | - | 3531.1 | - | auc=0.763032 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 45.9 | 45.9..45.9 | 1 | 1.391 | 1.372 | - | 3505.8 | - | auc=0.792792 | yes | UNKNOWN | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:5187 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

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

### iforest / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/iforest.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 62.0 | 62.0..62.0 | 1 | - | - | - | 823.9 | - | auc=0.542833 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 58.8 | 58.8..58.8 | 1 | - | - | - | 888.1 | - | auc=0.542833 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 45.3 | 45.3..45.3 | 1 | 1.368 | 1.296 | - | 893.8 | - | auc=0.538780 | yes | UNKNOWN | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:4660 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

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

### rf / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/rf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 706.1 | 706.1..706.1 | 1 | - | - | - | 3145.3 | - | auc=0.876094, logloss=0.444021 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 686.1 | 686.1..686.1 | 1 | - | - | - | 3566.2 | - | auc=0.876094, logloss=0.444021 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 70.2 | 70.2..70.2 | 1 | 10.062 | 9.777 | - | 3713.6 | - | auc=0.870452, logloss=0.486884 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 6922.3 | 6922.3..6922.3 | 1 | 0.102 | 0.099 | - | 10611.3 | - | auc=0.877515, logloss=0.262548 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:8755,ours:7559,ours-ab:7559,sklearn-rf-cpu:7363 spread=0.1590 verdict=NOT-COMPARABLE`

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
| max_depth | 16 | 16 | 16 | 16 |
| max_features | - | "sqrt" | "sqrt" | "sqrt" |
| max_leaves | 65536 | -1 | -1 | null |
| max_samples | - | 1.0 | 1.0 | 1.0 |
| min_child_weight | 0.0 | - | - | - |
| min_samples_leaf | 1 | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| reg_alpha | 0.0 | - | - | - |
| reg_lambda | 0.0 | - | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | 0.632 | - | - | - |

accepted difference: ours-ab class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

### rf / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/rf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1063.9 | 1063.9..1063.9 | 1 | - | - | - | 1139.1 | - | auc=0.578982, logloss=0.669411 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 992.3 | 992.3..992.3 | 1 | - | - | - | 1176.3 | - | auc=0.578982, logloss=0.669411 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 62.7 | 62.7..62.7 | 1 | 16.978 | 15.836 | - | 1106.9 | - | auc=0.571638, logloss=0.607659 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 12595.6 | 12595.6..12595.6 | 1 | 0.084 | 0.079 | - | 1557.6 | - | auc=0.578311, logloss=0.772297 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:17381,ours:33618,ours-ab:33618,sklearn-rf-cpu:32272 spread=0.4830 verdict=NOT-COMPARABLE`

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
| max_depth | 16 | 16 | 16 | 16 |
| max_features | - | "sqrt" | "sqrt" | "sqrt" |
| max_leaves | 65536 | -1 | -1 | null |
| max_samples | - | 1.0 | 1.0 | 1.0 |
| min_child_weight | 0.0 | - | - | - |
| min_samples_leaf | 1 | 1 | 1 | 1 |
| min_split_gain | 0.0 | 0.0 | 0.0 | 0.0 |
| n_estimators | 100 | 100 | 100 | 100 |
| reg_alpha | 0.0 | - | - | - |
| reg_lambda | 0.0 | - | - | - |
| seed | 7 | 7 | 7 | 7 |
| subsample | 0.632 | - | - | - |

accepted difference: ours-ab class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: sklearn-rf-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

accepted difference: lightgbm-cpu class_weight: None on every arm is unit class weights, the value each library defines for None

accepted difference: lightgbm-cpu max_leaves: no leaf cap on any arm: ours max_leaves -1 (cuML's sentinel), sklearn max_leaf_nodes None (a value would switch it to best-first growth), LightGBM num_leaves 65536 = 2 ** max_depth, never reached at depth 16

## Classical

### dbscan / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.dbscan.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 49.9 | 49.9..49.9 | 1 | - | - | - | 356.2 | - | n_clusters=18, noise_fraction=0.611000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 51.3 | 51.3..51.3 | 1 | - | - | - | 356.9 | - | ari_vs_ours=1.000000, n_clusters=18, noise_agreement_vs_ours=1.000000, noise_fraction=0.611000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 37.9 | 37.9..37.9 | 1 | 1.316 | 1.355 | - | 186.9 | - | ari_vs_ours=1.000000, n_clusters=18, noise_agreement_vs_ours=1.000000, noise_fraction=0.611000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "rbc" | "rbc" | "auto" |
| eps | 4.17 | 4.17 | 4.17 |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_samples | 10 | 10 | 10 |
| p | - | - | null |

accepted difference: ours seed: mojolearn DBSCAN has no seed argument (deterministic)

accepted difference: ours-fast seed: mojolearn DBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu seed: scikit-learn DBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu algorithm: an exact eps search on every arm: ours 'rbc' (its default), scikit-learn has no 'rbc' and runs 'auto' (a tree on taxi, brute on Istella-S)

### dbscan / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.dbscan.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 39.3 | 39.3..39.3 | 1 | - | - | - | 347.1 | - | n_clusters=11, noise_fraction=0.366000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 34.4 | 34.4..34.4 | 1 | - | - | - | 346.4 | - | ari_vs_ours=1.000000, n_clusters=11, noise_agreement_vs_ours=1.000000, noise_fraction=0.366000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 93.0 | 93.0..93.0 | 1 | 0.423 | 0.370 | - | 145.4 | - | ari_vs_ours=1.000000, n_clusters=11, noise_agreement_vs_ours=1.000000, noise_fraction=0.366000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "rbc" | "rbc" | "auto" |
| eps | 0.177 | 0.177 | 0.177 |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| min_samples | 10 | 10 | 10 |
| p | - | - | null |

accepted difference: ours seed: mojolearn DBSCAN has no seed argument (deterministic)

accepted difference: ours-fast seed: mojolearn DBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu seed: scikit-learn DBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu algorithm: an exact eps search on every arm: ours 'rbc' (its default), scikit-learn has no 'rbc' and runs 'auto' (a tree on taxi, brute on Istella-S)

### hdbscan / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 102.6 | 102.6..102.6 | 1 | - | - | - | 373.8 | - | n_clusters=5, noise_fraction=0.492500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 112.5 | 112.5..112.5 | 1 | - | - | - | 370.7 | - | ari_vs_ours=1.000000, n_clusters=5, noise_agreement_vs_ours=1.000000, noise_fraction=0.492500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 364.8 | 364.8..364.8 | 1 | 0.281 | 0.308 | - | 148.2 | - | ari_vs_ours=0.931372, n_clusters=5, noise_agreement_vs_ours=0.972500, noise_fraction=0.468000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

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

accepted difference: ours seed: mojolearn HDBSCAN has no seed argument (deterministic)

accepted difference: ours-fast seed: mojolearn HDBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu seed: scikit-learn HDBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu max_cluster_size: no limit on every arm: ours and cuML spell it 0, scikit-learn None

### hdbscan / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 76.8 | 76.8..76.8 | 1 | - | - | - | 366.7 | - | n_clusters=4, noise_fraction=0.211500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 101.2 | 101.2..101.2 | 1 | - | - | - | 364.6 | - | ari_vs_ours=1.000000, n_clusters=4, noise_agreement_vs_ours=1.000000, noise_fraction=0.211500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 107.3 | 107.3..107.3 | 1 | 0.716 | 0.943 | - | 138.6 | - | ari_vs_ours=0.991083, n_clusters=4, noise_agreement_vs_ours=0.994500, noise_fraction=0.206000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

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

accepted difference: ours seed: mojolearn HDBSCAN has no seed argument (deterministic)

accepted difference: ours-fast seed: mojolearn HDBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu seed: scikit-learn HDBSCAN has no seed argument (deterministic)

accepted difference: sklearn-cpu max_cluster_size: no limit on every arm: ours and cuML spell it 0, scikit-learn None

### kde / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.kde.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24.3 | 24.3..24.3 | 1 | - | - | - | 317.6 | - | mean_log_likelihood=-404.680084, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 51.3 | 51.3..51.3 | 1 | - | - | - | 316.3 | - | mean_log_likelihood=-404.680084, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 61.0 | 61.0..61.0 | 1 | 0.398 | 0.841 | - | 140.5 | - | mean_log_likelihood=-402.006921, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| atol | 0.0 | 0.0 | 0.0 |
| bandwidth | 0.9666366534954752 | 0.9666366534954752 | 0.9666366534954752 |
| breadth_first | true | true | true |
| kernel | "gaussian" | "gaussian" | "gaussian" |
| leaf_size | 40 | 40 | 40 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| rtol | 0.0 | 0.0 | 0.0 |

accepted difference: ours seed: mojolearn KernelDensity has no seed argument (exact density)

accepted difference: ours-fast seed: mojolearn KernelDensity has no seed argument (exact density)

accepted difference: sklearn-cpu seed: scikit-learn KernelDensity has no seed argument (exact density)

### kde / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.kde.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 18.9 | 18.9..18.9 | 1 | - | - | - | 314.5 | - | mean_log_likelihood=-11.508770, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18.2 | 18.2..18.2 | 1 | - | - | - | 312.3 | - | mean_log_likelihood=-11.508770, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12.9 | 12.9..12.9 | 1 | 1.471 | 1.415 | - | 135.1 | - | mean_log_likelihood=-11.508770, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "auto" | "auto" | "auto" |
| atol | 0.0 | 0.0 | 0.0 |
| bandwidth | 0.6024643228025249 | 0.6024643228025249 | 0.6024643228025249 |
| breadth_first | true | true | true |
| kernel | "gaussian" | "gaussian" | "gaussian" |
| leaf_size | 40 | 40 | 40 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| rtol | 0.0 | 0.0 | 0.0 |

accepted difference: ours seed: mojolearn KernelDensity has no seed argument (exact density)

accepted difference: ours-fast seed: mojolearn KernelDensity has no seed argument (exact density)

accepted difference: sklearn-cpu seed: scikit-learn KernelDensity has no seed argument (exact density)

### kmeans / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 51.1 | 51.1..51.1 | 1 | - | - | - | 323.5 | - | inertia=4.423e+14, inertia_over_ours=1.000000, n_iter=21 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46.4 | 46.4..46.4 | 1 | - | - | - | 323.7 | - | inertia=4.423e+14, inertia_over_ours=0.999996, n_iter=21 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 26.1 | 26.1..26.1 | 1 | 1.960 | 1.780 | - | 156.4 | - | inertia=4.423e+14, inertia_over_ours=1.000002, n_iter=20 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 25.8 | 25.8..25.8 | 1 | 1.977 | 1.795 | - | 482.8 | 50.5 | inertia=4.423e+14, inertia_over_ours=1.000005, n_iter=20 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | - | - | "lloyd" | - |
| init | "array sha256:aa04b3be26a0e683" | "array sha256:aa04b3be26a0e683" | "array sha256:aa04b3be26a0e683" | "array sha256:aa04b3be26a0e683" |
| max_iter | 20 | 20 | 20 | 20 |
| metric | "euclidean" | "euclidean" | - | "euclidean" |
| n_clusters | 64 | 64 | 64 | 64 |
| n_init | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 | - |

### kmeans / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 45.3 | 45.3..45.3 | 1 | - | - | - | 319.0 | - | inertia=6524.265344, inertia_over_ours=1.000000, n_iter=21 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 37.6 | 37.6..37.6 | 1 | - | - | - | 320.6 | - | inertia=6524.265344, inertia_over_ours=1.000000, n_iter=21 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 17.9 | 17.9..17.9 | 1 | 2.522 | 2.094 | - | 141.2 | - | inertia=6524.265571, inertia_over_ours=1.000000, n_iter=20 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 20.7 | 20.7..20.7 | 1 | 2.187 | 1.816 | - | 445.3 | 18.5 | inertia=6524.265588, inertia_over_ours=1.000000, n_iter=20 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | - | - | "lloyd" | - |
| init | "array sha256:d37dac3d07d6a88e" | "array sha256:d37dac3d07d6a88e" | "array sha256:d37dac3d07d6a88e" | "array sha256:d37dac3d07d6a88e" |
| max_iter | 20 | 20 | 20 | 20 |
| metric | "euclidean" | "euclidean" | - | "euclidean" |
| n_clusters | 64 | 64 | 64 | 64 |
| n_init | 1 | 1 | 1 | 1 |
| seed | 7 | 7 | 7 | 7 |
| tol | 1e-07 | 1e-07 | 1e-07 | - |

### knn / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25.2 | 25.2..25.2 | 1 | - | - | - | 319.4 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21.2 | 21.2..21.2 | 1 | - | - | - | 317.3 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 19.5 | 19.5..19.5 | 1 | 1.289 | 1.087 | - | 152.5 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 19.6 | 19.6..19.6 | 1 | 1.286 | 1.084 | - | 458.9 | 40.5 | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 | - |
| metric | "euclidean" | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 | 10 |
| p | 2 | 2 | 2 | 2 |
| seed | - | - | - | 7 |

accepted difference: ours seed: mojolearn NearestNeighbors has no seed argument (exact search)

accepted difference: ours-fast seed: mojolearn NearestNeighbors has no seed argument (exact search)

accepted difference: sklearn-cpu seed: scikit-learn NearestNeighbors has no seed argument (exact search)

### knn / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 30.2 | 30.2..30.2 | 1 | - | - | - | 346.4 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 23.5 | 23.5..23.5 | 1 | - | - | - | 346.2 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12.5 | 12.5..12.5 | 1 | 2.423 | 1.885 | - | 142.3 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 17.6 | 17.6..17.6 | 1 | 1.714 | 1.333 | - | 424.3 | 8.5 | recall_at_10=1.000000, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| algorithm | "brute" | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 | - |
| metric | "euclidean" | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 | 10 |
| p | 2 | 2 | 2 | 2 |
| seed | - | - | - | 7 |

accepted difference: ours seed: mojolearn NearestNeighbors has no seed argument (exact search)

accepted difference: ours-fast seed: mojolearn NearestNeighbors has no seed argument (exact search)

accepted difference: sklearn-cpu seed: scikit-learn NearestNeighbors has no seed argument (exact search)

### ols / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.ols.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 509.8 | 509.8..509.8 | 1 | - | - | - | 328.5 | - | finite=True, r2=-24.129541, rmse=4.746592 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 387.7 | 387.7..387.7 | 1 | - | - | - | 327.6 | - | finite=True, r2=-24.390212, rmse=4.771147 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 24.5 | 24.5..24.5 | 1 | 20.820 | 15.834 | - | 150.5 | - | finite=True, r2=-0.027117, rmse=0.959621 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| fit_intercept | true | true | true | true |
| positive | - | - | false | - |
| seed | - | - | - | 7 |
| tol | - | - | 1e-06 | - |

accepted difference: ours seed: mojolearn LinearRegression has no seed argument (closed-form fit)

accepted difference: ours-fast seed: mojolearn LinearRegression has no seed argument (closed-form fit)

accepted difference: sklearn-cpu seed: scikit-learn LinearRegression has no seed argument (closed-form fit)

### ols / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 30.2 | 30.2..30.2 | 1 | - | - | - | 313.3 | - | finite=True, r2=0.889788, rmse=6.579774 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.0 | 6.0..6.0 | 1 | - | - | - | 313.5 | - | finite=True, r2=0.889542, rmse=6.587113 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4.2 | 4.2..4.2 | 1 | 7.232 | 1.447 | - | 135.5 | - | finite=True, r2=0.907724, rmse=6.020627 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| fit_intercept | true | true | true | true |
| positive | - | - | false | - |
| seed | - | - | - | 7 |
| tol | - | - | 1e-06 | - |

accepted difference: ours seed: mojolearn LinearRegression has no seed argument (closed-form fit)

accepted difference: ours-fast seed: mojolearn LinearRegression has no seed argument (closed-form fit)

accepted difference: sklearn-cpu seed: scikit-learn LinearRegression has no seed argument (closed-form fit)

### pca / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 105.0 | 105.0..105.0 | 1 | - | - | - | 351.7 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 75.7 | 75.7..75.7 | 1 | - | - | - | 355.8 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 16.6 | 16.6..16.6 | 1 | 6.329 | 4.566 | - | 142.7 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::_linalg_eigh.eigenvalues' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on http) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| n_components | 8 | 8 | 8 | 8 |
| seed | 7 | 7 | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | 0.0 | 0.0 | - |
| whiten | false | false | false | false |

### pca / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22.9 | 22.9..22.9 | 1 | - | - | - | 344.8 | - | explained_variance_ratio_sum=0.999988 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.0 | 6.0..6.0 | 1 | - | - | - | 344.6 | - | explained_variance_ratio_sum=0.999988 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4.3 | 4.3..4.3 | 1 | 5.280 | 1.394 | - | 135.5 | - | explained_variance_ratio_sum=0.999987 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::_linalg_eigh.eigenvalues' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on http) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) | torch (declared) |
| n_components | 8 | 8 | 8 | 8 |
| seed | 7 | 7 | 7 | 7 |
| svd_solver | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" | "covariance_eigh" |
| tol | 0.0 | 0.0 | 0.0 | - |
| whiten | false | false | false | false |

### svc / istella (rows 2000, shape 256x220)

race: done, driver rc 0, log `logs/classical.svc.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 38.9 | 38.9..38.9 | 1 | - | - | - | 325.3 | - | accuracy=0.855469, n_support=94 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 31.8 | 31.8..31.8 | 1 | - | - | - | 322.3 | - | accuracy=0.855469, n_support=94 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 11.3 | 11.3..11.3 | 1 | 3.431 | 2.810 | - | 134.9 | - | accuracy=0.855469, n_support=94 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

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

### svc / taxi (rows 2000, shape 256x11)

race: done, driver rc 0, log `logs/classical.svc.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 36.0 | 36.0..36.0 | 1 | - | - | - | 324.0 | - | accuracy=0.765625, n_support=151 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 33.6 | 33.6..33.6 | 1 | - | - | - | 323.0 | - | accuracy=0.765625, n_support=151 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8.8 | 8.8..8.8 | 1 | 4.073 | 3.803 | - | 135.6 | - | accuracy=0.765625, n_support=151 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

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

## Classical, wave 2

### agglomerative / istella (rows 2000, shape X 2000x220)

race: done, driver rc 0, log `logs/classical2.agglomerative.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 41.0 | 41.0..41.0 | 1 | - | - | - | 369.0 | - | n_clusters=8, silhouette=0.652303 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 38.8 | 38.8..38.8 | 1 | - | - | - | 367.0 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.652303 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 195.9 | 195.9..195.9 | 1 | 0.209 | 0.198 | - | 142.4 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.652303 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows (the full board; this run caps them at --rows 2000): 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| linkage | "single" | "single" | "single" |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 | 8 |

accepted difference: ours seed: mojolearn AgglomerativeClustering has no seed argument

accepted difference: ours-fast seed: mojolearn AgglomerativeClustering has no seed argument

accepted difference: sklearn-cpu seed: scikit-learn AgglomerativeClustering has no seed argument

### agglomerative / taxi (rows 2000, shape X 2000x11)

race: done, driver rc 0, log `logs/classical2.agglomerative.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 28.4 | 28.4..28.4 | 1 | - | - | - | 361.9 | - | n_clusters=8, silhouette=0.522237 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 41.2 | 41.2..41.2 | 1 | - | - | - | 362.0 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.522237 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14.8 | 14.8..14.8 | 1 | 1.920 | 2.791 | - | 136.1 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.522237 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, linkage='single', metric='euclidean' (ours and cuML connectivity='pairwise'). Rows (the full board; this run caps them at --rows 2000): 10000 stride rows of the cls block (standardized by the fit rows); O(n^2). Timed: fit.

mismatch: single linkage only: ours and cuML implement no other linkage

mismatch: seed: no arm has a seed argument (deterministic)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | sklearn (get_params) |
| linkage | "single" | "single" | "single" |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_clusters | 8 | 8 | 8 |

accepted difference: ours seed: mojolearn AgglomerativeClustering has no seed argument

accepted difference: ours-fast seed: mojolearn AgglomerativeClustering has no seed argument

accepted difference: sklearn-cpu seed: scikit-learn AgglomerativeClustering has no seed argument

### arima / synthetic (rows 2000, shape Yfit 64x2000; Yhold 64x100)

race: done, driver rc 0, log `logs/classical2.arima.synthetic.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 96.0 | 96.0..96.0 | 1 | - | - | - | 575.8 | - | forecast_rmse=1.515518, insample_rmse=0.999342, mean_aic=5680.976967, mean_llf=-2836.488483 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 109.9 | 109.9..109.9 | 1 | - | - | - | 572.2 | - | forecast_rmse=1.515540, insample_rmse=0.999341, mean_aic=5680.971687, mean_llf=-2836.485844 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 301.3 | 301.3..301.3 | 1 | 0.319 | 0.365 | - | 142.9 | - | forecast_rmse=1.515423, insample_rmse=0.999338, mean_aic=5680.957160, mean_llf=-2836.478580 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: order=(1,0,1), seasonal_order=(0,0,0,0), trend='c' (cuML fit_intercept=True), maxiter=1000, maximum likelihood. Rows (the full board; this run caps them at --rows 2000): 64 synthetic ARMA(1,1) series, 2000 fit points, 100 held out. Timed: fit of every series.

mismatch: ours and cuML fit the whole batch in one call; statsmodels fits one series per call (the state-space model, L-BFGS), spread over every core with joblib

mismatch: statsmodels enforce_stationarity and enforce_invertibility at its default (True); ours and cuML have no such parameter

mismatch: seed: no arm has a seed argument (maximum likelihood)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | statsmodels (declared) |
| max_iter | 1000 | 1000 | 1000 |
| order | [1, 0, 1] | [1, 0, 1] | [1, 0, 1] |
| seasonal_order | [0, 0, 0, 0] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| trend | "c" | "c" | "c" |

accepted difference: ours seed: no ARIMA takes a seed argument (maximum likelihood, deterministic): ours, statsmodels, cuML

accepted difference: ours-fast seed: no ARIMA takes a seed argument (maximum likelihood, deterministic): ours, statsmodels, cuML

accepted difference: statsmodels-cpu seed: no ARIMA takes a seed argument (maximum likelihood, deterministic): ours, statsmodels, cuML

### elasticnet / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.elasticnet.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5104.4 | 5104.4..5104.4 | 1 | - | - | - | 371.0 | - | finite=True, r2=0.275634, rmse=0.675632 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 26.2 | 26.2..26.2 | 1 | - | - | - | 330.1 | - | finite=True, r2=0.275640, rmse=0.675630 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 20.7 | 20.7..20.7 | 1 | 246.014 | 1.263 | - | 145.1 | - | finite=True, r2=0.275634, rmse=0.675632 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 | 0.01 |
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

### elasticnet / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.elasticnet.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 24.9 | 24.9..24.9 | 1 | - | - | - | 334.9 | - | finite=True, r2=0.930720, rmse=3.844767 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.1 | 3.1..3.1 | 1 | - | - | - | 314.7 | - | finite=True, r2=0.930720, rmse=3.844768 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.6 | 0.6..0.6 | 1 | 38.730 | 4.893 | - | 135.4 | - | finite=True, r2=0.930720, rmse=3.844767 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.01 | 0.01 | 0.01 |
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

### ets / synthetic (rows 2000, shape Yfit 64x1440; Yhold 64x48)

race: done, driver rc 0, log `logs/classical2.ets.synthetic.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 221.5 | 221.5..221.5 | 1 | - | - | - | 337.6 | - | forecast_rmse=0.984392, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 196.9 | 196.9..196.9 | 1 | - | - | - | 339.0 | - | forecast_rmse=0.984473, insample_rmse=0.990971 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| statsmodels-cpu | statsmodels | cpu | opponent | 903.4 | 903.4..903.4 | 1 | 0.245 | 0.218 | - | 139.7 | - | forecast_rmse=0.984418, insample_rmse=0.991812 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, statsmodels-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: trend additive, seasonal additive, seasonal_periods=24, initialization_method='estimated'; ours and cuML start_periods=2, eps=2.24e-3; statsmodels damped_trend=False, use_boxcox=False. Rows (the full board; this run caps them at --rows 2000): 64 synthetic hourly series, period 24, 1440 fit points, 48 held out. Timed: construct + fit of every series.

mismatch: initialization: ours 'estimated' (its default, statsmodels' definition), statsmodels 'estimated'; cuML has only its heuristic start (start_periods=2), so its row fits the older initialization

mismatch: cuML returns no in-sample predictions; that quality cell is empty

mismatch: trend: ours and cuML are additive-trend with no parameter; statsmodels trend='additive'. eps is ours' and cuML's only; statsmodels fit() uses its own optimizer

mismatch: seed: no arm has a seed argument

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | statsmodels-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (attributes) | mojolearn (attributes) | statsmodels (declared) |
| damped_trend | - | - | false |
| eps | 0.00224 | 0.00224 | - |
| initialization_method | "estimated" | "estimated" | "estimated" |
| seasonal | "additive" | "additive" | "additive" |
| seasonal_periods | 24 | 24 | 24 |
| start_periods | 2 | 2 | - |
| trend | - | - | "additive" |

accepted difference: ours seed: no Holt-Winters takes a seed argument (deterministic): ours, statsmodels, cuML

accepted difference: ours-fast seed: no Holt-Winters takes a seed argument (deterministic): ours, statsmodels, cuML

accepted difference: statsmodels-cpu seed: no Holt-Winters takes a seed argument (deterministic): ours, statsmodels, cuML

### gmm / istella (rows 2000, shape X 2000x220; Xq 2000x220)

race: failed, driver rc 1, log `logs/classical2.gmm.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"GaussianMixture: fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to d) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"GaussianMixture: fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to d) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

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

### gmm / taxi (rows 2000, shape X 2000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 294.9 | 294.9..294.9 | 1 | - | - | - | 337.7 | - | bic=-68064.910372, mean_log_likelihood=12.918593, n_iter=25 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 88.9 | 88.9..88.9 | 1 | - | - | - | 325.3 | - | bic=-65102.837429, mean_log_likelihood=12.478869, n_iter=11 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 130.9 | 130.9..130.9 | 1 | 2.253 | 0.679 | - | 138.5 | - | bic=-69061.429868, mean_log_likelihood=13.245844, n_iter=100 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: init_params='kmeans': each library seeds its own k-means (ours the identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))

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

### gpc / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.gpc.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 337.0 | 337.0..337.0 | 1 | - | - | - | 545.5 | - | accuracy=0.919000, logloss=0.215232, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 556.1 | 556.1..556.1 | 1 | - | - | - | 538.1 | - | accuracy=0.919000, logloss=0.215260, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 456.0 | 456.0..456.0 | 1 | 0.739 | 1.219 | - | 408.4 | - | accuracy=0.919000, logloss=0.215253, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)), optimizer=None, max_iter_predict=100, n_restarts_optimizer=0. Rows (the full board; this run caps them at --rows 2000): 3000 fit and 3000 held-out stride rows of the cls block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: seed: ours refuses random_state (optimizer=None draws nothing); scikit-learn random_state=7

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

### gpc / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.gpc.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 327.4 | 327.4..327.4 | 1 | - | - | - | 530.5 | - | accuracy=0.764000, logloss=0.541835, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 358.5 | 358.5..358.5 | 1 | - | - | - | 518.8 | - | accuracy=0.764000, logloss=0.541865, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 246.2 | 246.2..246.2 | 1 | 1.330 | 1.456 | - | 402.7 | - | accuracy=0.764000, logloss=0.541862, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)), optimizer=None, max_iter_predict=100, n_restarts_optimizer=0. Rows (the full board; this run caps them at --rows 2000): 3000 fit and 3000 held-out stride rows of the cls block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: seed: ours refuses random_state (optimizer=None draws nothing); scikit-learn random_state=7

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

### gpr / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.gpr.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 111.8 | 111.8..111.8 | 1 | - | - | - | 520.3 | - | finite=True, mean_log_predictive_density=-5.391488, r2=0.228689, rmse=0.697182 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 138.9 | 138.9..138.9 | 1 | - | - | - | 511.2 | - | finite=True, mean_log_predictive_density=-5.391598, r2=0.228699, rmse=0.697178 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 204.3 | 204.3..204.3 | 1 | 0.547 | 0.680 | - | 347.6 | - | finite=True, mean_log_predictive_density=-5.391831, r2=0.228693, rmse=0.697180 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)) + WhiteKernel(noise_level=0.01), alpha=2**-20 (the one ridge IDENTICAL accepts beside 0; the noise lives in the WhiteKernel so the float32 factor of K exists on taxi's near-duplicate rows), optimizer=None, normalize_y=False, n_restarts_optimizer=0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 3000 fit and 3000 held-out stride rows of the reg block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: kernel: the same kernel built from each library's own classes

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 9.5367431640625e-07 | 9.5367431640625e-07 | 9.5367431640625e-07 |
| kernel | "((ConstantKernel(1.0) * RBF([14.832396974191326])) + WhiteKernel(0.01))" | "((ConstantKernel(1.0) * RBF([14.832396974191326])) + WhiteKernel(0.01))" | "1**2 * RBF(length_scale=14.8) + WhiteKernel(noise_level=0.01)" |
| n_restarts_optimizer | 0 | 0 | 0 |
| seed | 7 | 7 | 7 |

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### gpr / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.gpr.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 98.6 | 98.6..98.6 | 1 | - | - | - | 524.1 | - | finite=True, mean_log_predictive_density=-279.852141, r2=0.838375, rmse=5.872471 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 125.8 | 125.8..125.8 | 1 | - | - | - | 498.4 | - | finite=True, mean_log_predictive_density=-279.853051, r2=0.838374, rmse=5.872494 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 48.1 | 48.1..48.1 | 1 | 2.049 | 2.612 | - | 340.1 | - | finite=True, mean_log_predictive_density=-279.891365, r2=0.838375, rmse=5.872469 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)) + WhiteKernel(noise_level=0.01), alpha=2**-20 (the one ridge IDENTICAL accepts beside 0; the noise lives in the WhiteKernel so the float32 factor of K exists on taxi's near-duplicate rows), optimizer=None, normalize_y=False, n_restarts_optimizer=0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 3000 fit and 3000 held-out stride rows of the reg block (standardized by the fit rows); O(n^3). Timed: fit.

mismatch: kernel: the same kernel built from each library's own classes

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 9.5367431640625e-07 | 9.5367431640625e-07 | 9.5367431640625e-07 |
| kernel | "((ConstantKernel(1.0) * RBF([3.3166247903554])) + WhiteKernel(0.01))" | "((ConstantKernel(1.0) * RBF([3.3166247903554])) + WhiteKernel(0.01))" | "1**2 * RBF(length_scale=3.32) + WhiteKernel(noise_level=0.01)" |
| n_restarts_optimizer | 0 | 0 | 0 |
| seed | 7 | 7 | 7 |

accepted difference: sklearn-cpu kernel: the same ConstantKernel(1.0) * RBF(sqrt(d)) (gpr: + WhiteKernel(1e-2)) built from each library's own kernel classes; their reprs differ

### ivf / istella (rows 2000, shape index 2048x220; queries 64x220)

race: done, driver rc 0, log `logs/classical2.ivf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 259.7 | 259.7..259.7 | 1 | - | - | - | 371.4 | - | recall_at_10=0.998437, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 139.0 | 139.0..139.0 | 1 | - | - | - | 375.0 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 14.1 | 14.1..14.1 | 1 | 18.475 | 9.887 | - | 58.5 | - | recall_at_10=1.000000, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows (the full board; this run caps them at --rows 2000): the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (get_params) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 | 10 |
| nlist | 512 | 512 | 512 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

### ivf / taxi (rows 2000, shape index 2048x11; queries 64x11)

race: done, driver rc 0, log `logs/classical2.ivf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 136.2 | 136.2..136.2 | 1 | - | - | - | 357.5 | - | recall_at_10=0.956250, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 150.3 | 150.3..150.3 | 1 | - | - | - | 358.3 | - | recall_at_10=0.956250, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| faiss-cpu | faiss | cpu | opponent | 11.2 | 11.2..11.2 | 1 | 12.171 | 13.424 | - | 40.6 | - | recall_at_10=0.903125, rows_with_repeated_ids=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, faiss-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: IVF-Flat, n_lists=1024, n_probes=32, k=10, squared L2, k-means 20 iterations, seed 7. Rows (the full board; this run caps them at --rows 2000): the classical knn lane's block (tools/knn_datasets.real_block): 400000 index rows, 4000 queries, raw. Timed: build + search of every query.

mismatch: quantizer training set: each library's own (FAISS subsamples to 256 rows per list; cuVS kmeans_trainset_fraction 0.5; ours its own)

mismatch: seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | faiss-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | faiss (declared) | mojolearn (get_params) | mojolearn (get_params) |
| metric | "sqeuclidean" | "sqeuclidean" | "sqeuclidean" |
| n_neighbors | 10 | 10 | 10 |
| nlist | 512 | 512 | 512 |
| nprobe | 32 | 32 | 32 |
| seed | 7 | 7 | 7 |

### kernel-ridge / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 41.4 | 41.4..41.4 | 1 | - | - | - | 383.2 | - | finite=True, r2=0.299806, rmse=0.664264 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 64.4 | 64.4..64.4 | 1 | - | - | - | 360.4 | - | finite=True, r2=0.299806, rmse=0.664264 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 30.5 | 30.5..30.5 | 1 | 1.355 | 2.108 | - | 228.9 | - | finite=True, r2=0.299806, rmse=0.664264 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows (the full board; this run caps them at --rows 2000): 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 | 1.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |

accepted difference: ours seed: no KernelRidge takes a seed argument (ours, scikit-learn, cuML)

accepted difference: ours-fast seed: no KernelRidge takes a seed argument (ours, scikit-learn, cuML)

accepted difference: sklearn-cpu seed: no KernelRidge takes a seed argument (ours, scikit-learn, cuML)

### kernel-ridge / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.kernel-ridge.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 38.2 | 38.2..38.2 | 1 | - | - | - | 566.8 | - | finite=True, r2=0.650912, rmse=8.630456 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 61.3 | 61.3..61.3 | 1 | - | - | - | 347.4 | - | finite=True, r2=0.650912, rmse=8.630457 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 25.5 | 25.5..25.5 | 1 | 1.496 | 2.399 | - | 203.9 | - | finite=True, r2=0.650912, rmse=8.630457 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0. Rows (the full board; this run caps them at --rows 2000): 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: seed: no arm has a seed argument (closed-form fit)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| coef0 | 1.0 | 1.0 | 1.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |

accepted difference: ours seed: no KernelRidge takes a seed argument (ours, scikit-learn, cuML)

accepted difference: ours-fast seed: no KernelRidge takes a seed argument (ours, scikit-learn, cuML)

accepted difference: sklearn-cpu seed: no KernelRidge takes a seed argument (ours, scikit-learn, cuML)

### knn-clf / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.knn-clf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15.9 | 15.9..15.9 | 1 | - | - | - | 335.1 | - | accuracy=0.911000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 21.2 | 21.2..21.2 | 1 | - | - | - | 327.4 | - | accuracy=0.911000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28.2 | 28.2..28.2 | 1 | 0.564 | 0.750 | - | 185.9 | - | accuracy=0.911000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows (the full board; this run caps them at --rows 2000): 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| weights | "uniform" | "uniform" | "uniform" |

accepted difference: ours seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: ours-fast seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: sklearn-cpu seed: scikit-learn KNeighbors* has no seed argument (exact search)

### knn-clf / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.knn-clf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.7 | 13.7..13.7 | 1 | - | - | - | 350.7 | - | accuracy=0.727500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13.1 | 13.1..13.1 | 1 | - | - | - | 347.2 | - | accuracy=0.727500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 25.6 | 25.6..25.6 | 1 | 0.535 | 0.510 | - | 145.7 | - | accuracy=0.727500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows (the full board; this run caps them at --rows 2000): 200000 fit rows, 4000 queries (stride subsets of the cls block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| weights | "uniform" | "uniform" | "uniform" |

accepted difference: ours seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: ours-fast seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: sklearn-cpu seed: scikit-learn KNeighbors* has no seed argument (exact search)

### knn-reg / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.knn-reg.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10.0 | 10.0..10.0 | 1 | - | - | - | 333.6 | - | finite=True, r2=0.239429, rmse=0.692311 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 15.2 | 15.2..15.2 | 1 | - | - | - | 324.0 | - | finite=True, r2=0.239429, rmse=0.692311 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 26.8 | 26.8..26.8 | 1 | 0.372 | 0.568 | - | 185.1 | - | finite=True, r2=0.239429, rmse=0.692311 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows (the full board; this run caps them at --rows 2000): 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| weights | "uniform" | "uniform" | "uniform" |

accepted difference: ours seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: ours-fast seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: sklearn-cpu seed: scikit-learn KNeighbors* has no seed argument (exact search)

### knn-reg / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.knn-reg.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9.3 | 9.3..9.3 | 1 | - | - | - | 351.0 | - | finite=True, r2=0.860399, rmse=5.457721 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8.1 | 8.1..8.1 | 1 | - | - | - | 345.8 | - | finite=True, r2=0.860399, rmse=5.457721 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14.0 | 14.0..14.0 | 1 | 0.667 | 0.578 | - | 144.6 | - | finite=True, r2=0.860399, rmse=5.457721 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'. Rows (the full board; this run caps them at --rows 2000): 200000 fit rows, 4000 queries (stride subsets of the reg block, standardized by the fit rows). Timed: fit + predict of the queries.

mismatch: seed: no arm has a seed argument (exact search)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| leaf_size | - | - | 30 |
| metric | "euclidean" | "euclidean" | "euclidean" |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| weights | "uniform" | "uniform" | "uniform" |

accepted difference: ours seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: ours-fast seed: mojolearn KNeighbors* has no seed argument (exact search)

accepted difference: sklearn-cpu seed: scikit-learn KNeighbors* has no seed argument (exact search)

### lasso / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.lasso.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1333.5 | 1333.5..1333.5 | 1 | - | - | - | 357.5 | - | finite=True, r2=0.276321, rmse=0.675312 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12.6 | 12.6..12.6 | 1 | - | - | - | 328.4 | - | finite=True, r2=0.276322, rmse=0.675312 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 5.9 | 5.9..5.9 | 1 | 225.025 | 2.132 | - | 140.2 | - | finite=True, r2=0.276321, rmse=0.675312 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

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

### lasso / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.lasso.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 25.8 | 25.8..25.8 | 1 | - | - | - | 326.0 | - | finite=True, r2=0.930249, rmse=3.857817 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.2 | 3.2..3.2 | 1 | - | - | - | 315.2 | - | finite=True, r2=0.930249, rmse=3.857818 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.6 | 0.6..0.6 | 1 | 40.160 | 4.969 | - | 133.3 | - | finite=True, r2=0.930249, rmse=3.857817 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; ours and cuML solver='cd'. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: seed: ours refuses random_state (it selects nothing with selection='cyclic'), cuML has none; scikit-learn random_state=7

mismatch: tol: each library's own stopping rule reads it

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

### linearsvc / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.linearsvc.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 161.8 | 161.8..161.8 | 1 | - | - | - | 320.9 | - | accuracy=0.913500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 184.5 | 184.5..184.5 | 1 | - | - | - | 321.6 | - | accuracy=0.912500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 785.1 | 785.1..785.1 | 1 | 0.206 | 0.235 | - | 143.9 | - | accuracy=0.913500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

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
| seed | - | - | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn LinearSVC has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast seed: mojolearn LinearSVC has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

### linearsvc / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.linearsvc.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9.6 | 9.6..9.6 | 1 | - | - | - | 315.5 | - | accuracy=0.763000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7.5 | 7.5..7.5 | 1 | - | - | - | 316.4 | - | accuracy=0.763000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.2 | 3.2..3.2 | 1 | 3.008 | 2.340 | - | 135.2 | - | accuracy=0.762500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; scikit-learn liblinear (dual='auto'), which penalizes the intercept

mismatch: seed: ours and cuML LinearSVC have no seed argument

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
| seed | - | - | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn LinearSVC has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast seed: mojolearn LinearSVC has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

### linearsvr / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.linearsvr.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21.3 | 21.3..21.3 | 1 | - | - | - | 320.2 | - | finite=True, r2=-0.092549, rmse=0.829759 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 33.7 | 33.7..33.7 | 1 | - | - | - | 320.1 | - | finite=True, r2=-0.092549, rmse=0.829759 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 520.1 | 520.1..520.1 | 1 | 0.041 | 0.065 | - | 143.6 | - | finite=True, r2=-0.087102, rmse=0.827688 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

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
| seed | - | - | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn LinearSVR has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast seed: mojolearn LinearSVR has no seed argument (L-BFGS, deterministic)

### linearsvr / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.linearsvr.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 45.8 | 45.8..45.8 | 1 | - | - | - | 315.0 | - | finite=True, r2=0.461132, rmse=10.722794 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 38.7 | 38.7..38.7 | 1 | - | - | - | 314.6 | - | finite=True, r2=-0.420609, rmse=17.410203 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10.1 | 10.1..10.1 | 1 | 4.533 | 3.835 | - | 135.1 | - | finite=True, r2=0.923553, rmse=4.038760 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn has only l2

mismatch: solver: ours and cuML L-BFGS on the primal; scikit-learn liblinear dual coordinate descent (dual=True, the only form for this loss)

mismatch: intercept: ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0 (liblinear penalizes it)

mismatch: seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7

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
| seed | - | - | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn LinearSVR has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast seed: mojolearn LinearSVR has no seed argument (L-BFGS, deterministic)

### logreg / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.logreg.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 148.7 | 148.7..148.7 | 1 | - | - | - | 321.4 | - | accuracy=0.923500, logloss=0.204798, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 154.0 | 154.0..154.0 | 1 | - | - | - | 322.5 | - | accuracy=0.923500, logloss=0.204812, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 18.3 | 18.3..18.3 | 1 | 8.136 | 8.426 | - | 137.2 | - | accuracy=0.923500, logloss=0.204942, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

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
| seed | - | - | 7 |
| solver | "qn" | "qn" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn LogisticRegression has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast seed: mojolearn LogisticRegression has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: ours-fast l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu solver: L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'

### logreg / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.logreg.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.7 | 13.7..13.7 | 1 | - | - | - | 316.2 | - | accuracy=0.763500, logloss=0.547874, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 14.8 | 14.8..14.8 | 1 | - | - | - | 316.0 | - | accuracy=0.763500, logloss=0.547874, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.8 | 1.8..1.8 | 1 | 7.592 | 8.161 | - | 135.5 | - | accuracy=0.763500, logloss=0.547867, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; each library's own stopping rule reads tol

mismatch: seed: ours and cuML LogisticRegression have no seed argument

mismatch: l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

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
| seed | - | - | 7 |
| solver | "qn" | "qn" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours seed: mojolearn LogisticRegression has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast seed: mojolearn LogisticRegression has no seed argument (L-BFGS, deterministic)

accepted difference: ours-fast class_weight: None (unweighted) set explicitly on every arm

accepted difference: ours-fast l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu class_weight: None (unweighted) set explicitly on every arm

accepted difference: sklearn-cpu l1_ratio: penalty='l2' on every arm: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)

accepted difference: sklearn-cpu solver: L-BFGS on every arm: ours and cuML 'qn', scikit-learn 'lbfgs'

### nystroem / istella (rows 2000, shape X 2000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.nystroem.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 501.8 | 501.8..501.8 | 1 | - | - | - | 367.2 | - | kernel_rel_error=0.028603 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 389.0 | 389.0..389.0 | 1 | - | - | - | 368.0 | - | kernel_rel_error=0.028603 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 185.6 | 185.6..185.6 | 1 | 2.704 | 2.096 | - | 224.2 | - | kernel_rel_error=0.028046 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, n_components=256, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the landmark rows are each library's own random draw from seed 7

mismatch: degree=3, coef0=1.0 on every arm; the rbf kernel reads neither

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

### nystroem / taxi (rows 2000, shape X 2000x11; Xcheck 1000x11)

race: done, driver rc 0, log `logs/classical2.nystroem.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 617.6 | 617.6..617.6 | 1 | - | - | - | 520.1 | - | kernel_rel_error=0.017836 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 472.9 | 472.9..472.9 | 1 | - | - | - | 359.0 | - | kernel_rel_error=0.017836 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 198.7 | 198.7..198.7 | 1 | 3.108 | 2.380 | - | 160.0 | - | kernel_rel_error=0.021057 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, n_components=256, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the landmark rows are each library's own random draw from seed 7

mismatch: degree=3, coef0=1.0 on every arm; the rbf kernel reads neither

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

### rbf-sampler / istella (rows 2000, shape X 2000x220; Xcheck 1000x220)

race: done, driver rc 0, log `logs/classical2.rbf-sampler.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.0 | 5.0..5.0 | 1 | - | - | - | 331.5 | - | kernel_rel_error=0.135214 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.9 | 5.9..5.9 | 1 | - | - | - | 329.1 | - | kernel_rel_error=0.135214 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.1 | 2.1..2.1 | 1 | 2.409 | 2.845 | - | 146.8 | - | kernel_rel_error=0.139828 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: gamma=1/d, n_components=256, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the random Fourier features are each library's own draw from seed 7

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### rbf-sampler / taxi (rows 2000, shape X 2000x11; Xcheck 1000x11)

race: done, driver rc 0, log `logs/classical2.rbf-sampler.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.1 | 4.1..4.1 | 1 | - | - | - | 513.8 | - | kernel_rel_error=0.134474 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.7 | 4.7..4.7 | 1 | - | - | - | 320.5 | - | kernel_rel_error=0.134474 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.1 | 1.1..1.1 | 1 | 3.562 | 4.065 | - | 140.3 | - | kernel_rel_error=0.130369 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: gamma=1/d, n_components=256, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 stride rows of the reg block (standardized by the fit rows); kernel check on the first 1000. Timed: fit_transform of every row.

mismatch: the random Fourier features are each library's own draw from seed 7

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### ridge / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.ridge.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 448.1 | 448.1..448.1 | 1 | - | - | - | 326.3 | - | finite=True, r2=0.015427, rmse=0.787690 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 356.6 | 356.6..356.6 | 1 | - | - | - | 324.9 | - | finite=True, r2=0.015703, rmse=0.787580 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.1 | 1.1..1.1 | 1 | 409.093 | 325.556 | - | 142.7 | - | finite=True, r2=0.016246, rmse=0.787363 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| fit_intercept | true | true | true |
| max_iter | - | - | null |
| normalize | false | false | - |
| positive | - | - | false |
| seed | - | - | 7 |
| solver | "eig" | "eig" | "cholesky" |
| tol | - | - | 0.0001 |

accepted difference: ours seed: mojolearn Ridge has no seed argument (closed-form fit)

accepted difference: ours-fast seed: mojolearn Ridge has no seed argument (closed-form fit)

accepted difference: sklearn-cpu solver: ours and cuML 'eig' (eigendecomposition of the normal equations); scikit-learn has no 'eig' and runs 'cholesky' on the same normal equations

### ridge / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.ridge.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5.5 | 5.5..5.5 | 1 | - | - | - | 313.3 | - | finite=True, r2=0.930198, rmse=3.859240 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9.1 | 9.1..9.1 | 1 | - | - | - | 314.3 | - | finite=True, r2=0.930198, rmse=3.859238 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.6 | 0.6..0.6 | 1 | 9.612 | 15.930 | - | 136.0 | - | finite=True, r2=0.930198, rmse=3.859233 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

mismatch: solver, named on every arm: ours and cuML 'eig' (eigendecomposition of the normal equations), scikit-learn 'cholesky' (it has no 'eig')

mismatch: seed: ours and cuML Ridge have no seed argument

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| fit_intercept | true | true | true |
| max_iter | - | - | null |
| normalize | false | false | - |
| positive | - | - | false |
| seed | - | - | 7 |
| solver | "eig" | "eig" | "cholesky" |
| tol | - | - | 0.0001 |

accepted difference: ours seed: mojolearn Ridge has no seed argument (closed-form fit)

accepted difference: ours-fast seed: mojolearn Ridge has no seed argument (closed-form fit)

accepted difference: sklearn-cpu solver: ours and cuML 'eig' (eigendecomposition of the normal equations); scikit-learn has no 'eig' and runs 'cholesky' on the same normal equations

### spectral-embedding / istella (rows 2000, shape X 2000x220)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 58.9 | 58.9..58.9 | 1 | - | - | - | 357.1 | - | trustworthiness_k15=0.858681 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 45.9 | 45.9..45.9 | 1 | - | - | - | 354.0 | - | trustworthiness_k15=0.858681 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 105.5 | 105.5..105.5 | 1 | 0.559 | 0.435 | - | 179.1 | - | trustworthiness_k15=0.858691 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows (the full board; this run caps them at --rows 2000): the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

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

### spectral-embedding / taxi (rows 2000, shape X 2000x11)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 53.6 | 53.6..53.6 | 1 | - | - | - | 381.7 | - | trustworthiness_k15=0.690399 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 43.2 | 43.2..43.2 | 1 | - | - | - | 378.8 | - | trustworthiness_k15=0.708797 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 103.2 | 103.2..103.2 | 1 | 0.519 | 0.419 | - | 139.2 | - | trustworthiness_k15=0.687460 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows (the full board; this run caps them at --rows 2000): the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

mismatch: eigensolver: ours Lanczos (default tolerance); scikit-learn arpack (its default); cuML its own

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

### spectral / istella (rows 2000, shape X 2000x220)

race: done, driver rc 0, log `logs/classical2.spectral.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 177.5 | 177.5..177.5 | 1 | - | - | - | 360.9 | - | n_clusters=8, silhouette=0.058626 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 177.1 | 177.1..177.1 | 1 | - | - | - | 362.0 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.058626 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 167.9 | 167.9..167.9 | 1 | 1.058 | 1.055 | - | 172.7 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.058626 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10, n_components=8, random_state=7. Rows (the full board; this run caps them at --rows 2000): 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

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
| n_init | 10 | 10 | 10 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### spectral / taxi (rows 2000, shape X 2000x11)

race: done, driver rc 0, log `logs/classical2.spectral.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 128.8 | 128.8..128.8 | 1 | - | - | - | 388.8 | - | n_clusters=8, silhouette=0.084290 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 121.3 | 121.3..121.3 | 1 | - | - | - | 384.9 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.084290 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 153.1 | 153.1..153.1 | 1 | 0.841 | 0.792 | - | 148.1 | - | ari_vs_ours=0.993023, n_clusters=8, silhouette=0.083875 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10, n_components=8, random_state=7. Rows (the full board; this run caps them at --rows 2000): 10000 stride rows of the cls block (standardized by the fit rows); O(n^2) affinity. Timed: fit.

mismatch: eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn arpack eigen_tol='auto'; each library's own k-means on the embedding

mismatch: gamma, degree, coef0: not read by the nearest_neighbors affinity; ours refuses any value (None), scikit-learn holds 1.0, 3, 1

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

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
| n_init | 10 | 10 | 10 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |

accepted difference: ours-fast gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

accepted difference: sklearn-cpu gamma: affinity='nearest_neighbors' reads no gamma: ours refuses any value (None), scikit-learn holds its default (1.0 clustering, None embedding)

### svr / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.svr.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 30.0 | 30.0..30.0 | 1 | - | - | - | 351.9 | - | finite=True, r2=0.235325, rmse=0.694177 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 38.0 | 38.0..38.0 | 1 | - | - | - | 352.3 | - | finite=True, r2=0.235309, rmse=0.694184 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 68.7 | 68.7..68.7 | 1 | 0.436 | 0.553 | - | 149.6 | - | finite=True, r2=0.235295, rmse=0.694190 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows (the full board; this run caps them at --rows 2000): 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

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
| tol | 0.001 | 0.001 | 0.001 |

accepted difference: ours seed: no SVR takes a seed argument (ours, scikit-learn, cuML)

accepted difference: ours-fast seed: no SVR takes a seed argument (ours, scikit-learn, cuML)

accepted difference: sklearn-cpu seed: no SVR takes a seed argument (ours, scikit-learn, cuML)

### svr / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.svr.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 29.6 | 29.6..29.6 | 1 | - | - | - | 528.5 | - | finite=True, r2=0.654427, rmse=8.586900 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 30.2 | 30.2..30.2 | 1 | - | - | - | 333.3 | - | finite=True, r2=0.654429, rmse=8.586880 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 51.1 | 51.1..51.1 | 1 | 0.578 | 0.590 | - | 147.8 | - | finite=True, r2=0.654426, rmse=8.586920 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB. Rows (the full board; this run caps them at --rows 2000): 10000 fit and 10000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

mismatch: cache_size=2000 on every arm; ours honors it at predict only (DEVIATION 871); libsvm is single-threaded; shrinking=True is scikit-learn's only

mismatch: seed: no arm has a seed argument

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
| tol | 0.001 | 0.001 | 0.001 |

accepted difference: ours seed: no SVR takes a seed argument (ours, scikit-learn, cuML)

accepted difference: ours-fast seed: no SVR takes a seed argument (ours, scikit-learn, cuML)

accepted difference: sklearn-cpu seed: no SVR takes a seed argument (ours, scikit-learn, cuML)

### tsvd / istella (rows 2000, shape X 2000x220)

race: done, driver rc 0, log `logs/classical2.tsvd.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 78.4 | 78.4..78.4 | 1 | - | - | - | 352.4 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001106 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 63.2 | 63.2..63.2 | 1 | - | - | - | 352.8 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001106 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.4 | 2.4..2.4 | 1 | 32.584 | 26.294 | - | 137.1 | - | explained_variance_ratio_sum=1.000000, relative_reconstruction_error=0.0001083 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=8, tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "covariance_eigh" | "covariance_eigh" | "arpack" |
| n_components | 8 | 8 | 8 |
| n_iter | 5 | 5 | 5 |
| seed | 7 | 7 | 7 |
| tol | 0.0 | 0.0 | 0.0 |

accepted difference: sklearn-cpu algorithm: scikit-learn TruncatedSVD has no 'covariance_eigh'; it runs 'arpack' at tol=0

### tsvd / taxi (rows 2000, shape X 2000x11)

race: done, driver rc 0, log `logs/classical2.tsvd.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4.4 | 4.4..4.4 | 1 | - | - | - | 346.0 | - | explained_variance_ratio_sum=0.997435, relative_reconstruction_error=0.026699 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.4 | 3.4..3.4 | 1 | - | - | - | 344.7 | - | explained_variance_ratio_sum=0.997435, relative_reconstruction_error=0.026699 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.3 | 1.3..1.3 | 1 | 3.281 | 2.533 | - | 133.3 | - | explained_variance_ratio_sum=0.997434, relative_reconstruction_error=0.026699 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=8, tol=0.0, n_iter=5, n_oversamples=10, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 stride rows of the train split, raw (sentinel cleaned, not scaled). Timed: fit.

mismatch: algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn 'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "covariance_eigh" | "covariance_eigh" | "arpack" |
| n_components | 8 | 8 | 8 |
| n_iter | 5 | 5 | 5 |
| seed | 7 | 7 | 7 |
| tol | 0.0 | 0.0 | 0.0 |

accepted difference: sklearn-cpu algorithm: scikit-learn TruncatedSVD has no 'covariance_eigh'; it runs 'arpack' at tol=0

### umap / istella (rows 2000, shape X 2000x220)

race: done, driver rc 0, log `logs/classical2.umap.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 88.4 | 88.4..88.4 | 1 | - | - | - | 378.7 | - | trustworthiness_k15=0.943151 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 129.8 | 129.8..129.8 | 1 | - | - | - | 379.6 | - | trustworthiness_k15=0.949116 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 1737.5 | 1737.5..1737.5 | 1 | 0.051 | 0.075 | - | 402.0 | - | trustworthiness_k15=0.945949 | - | LIKE-FOR-LIKE-SPAN | - | ok |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1071.0 | 1071.0..1071.0 | 1 | 0.083 | 0.121 | - | 423.4 | - | trustworthiness_k15=0.944117 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0, n_epochs=200, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

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
| n_epochs | 200 | 200 | 200 | 200 |
| n_neighbors | 15 | 15 | 15 | 15 |
| negative_sample_rate | 5 | 5 | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 | 1.0 | 1.0 |
| seed | 7 | 7 | 7 | null |
| set_op_mix_ratio | 1.0 | 1.0 | 1.0 | 1.0 |
| spread | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: umap-learn-cpu-unseeded seed: raced unseeded on purpose: seeded umap-learn runs one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)

### umap / taxi (rows 2000, shape X 2000x11)

race: done, driver rc 0, log `logs/classical2.umap.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 74.7 | 74.7..74.7 | 1 | - | - | - | 387.0 | - | trustworthiness_k15=0.969395 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 99.7 | 99.7..99.7 | 1 | - | - | - | 387.6 | - | trustworthiness_k15=0.973998 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 1663.6 | 1663.6..1663.6 | 1 | 0.045 | 0.060 | - | 485.1 | - | trustworthiness_k15=0.973838 | - | LIKE-FOR-LIKE-SPAN | - | ok |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1029.0 | 1029.0..1029.0 | 1 | 0.073 | 0.097 | - | 502.4 | - | trustworthiness_k15=0.973131 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0, n_epochs=200, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

mismatch: neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); cuML build_algo='brute_force_knn' (exact)

mismatch: umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)

mismatch: umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; the seed is the one parameter that differs

mismatch: spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most 20 basis vectors)

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
| n_epochs | 200 | 200 | 200 | 200 |
| n_neighbors | 15 | 15 | 15 | 15 |
| negative_sample_rate | 5 | 5 | 5 | 5 |
| repulsion_strength | 1.0 | 1.0 | 1.0 | 1.0 |
| seed | 7 | 7 | 7 | null |
| set_op_mix_ratio | 1.0 | 1.0 | 1.0 | 1.0 |
| spread | 1.0 | 1.0 | 1.0 | 1.0 |

accepted difference: umap-learn-cpu-unseeded seed: raced unseeded on purpose: seeded umap-learn runs one thread (its rule); this arm is random_state=None, n_jobs=-1 (umap-learn-cpu has 7)

## Neural

### gemm / gaussian (neural shape small: 256x256x256)

race: failed, driver rc 3, log `logs/neural.gemm.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### lm-forward / bytes (neural shape small: B2 L64 DM64 H4 KV2 HD16 FF128 layers2 V256)

race: failed, driver rc 3, log `logs/neural.lm-forward.bytes.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### lm-train-step / bytes (neural shape small: B2 L64 DM64 H4 KV2 HD16 FF128 layers2 V256)

race: failed, driver rc 3, log `logs/neural.lm-train-step.bytes.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | - | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mamba1-forward / gaussian (neural shape small: B2 L64 DM16)

race: failed, driver rc 3, log `logs/neural.mamba1-forward.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mamba1-infer / gaussian (neural shape small: B2 L64 DM16)

race: failed, driver rc 3, log `logs/neural.mamba1-infer.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mamba2-forward / gaussian (neural shape small: B2 L64 DM64)

race: failed, driver rc 3, log `logs/neural.mamba2-forward.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mamba2-infer / gaussian (neural shape small: B2 L64 DM64)

race: failed, driver rc 3, log `logs/neural.mamba2-infer.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mamba3-forward / gaussian (neural shape small: B2 L64 DM64)

race: failed, driver rc 3, log `logs/neural.mamba3-forward.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mamba3-infer / gaussian (neural shape small: B2 L64 DM64)

race: failed, driver rc 3, log `logs/neural.mamba3-infer.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mlp-infer / gaussian (neural shape small: rows32 8-16-3)

race: failed, driver rc 3, log `logs/neural.mlp-infer.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mlp-train-step / gaussian (neural shape small: rows32 8-16-3)

race: failed, driver rc 3, log `logs/neural.mlp-train-step.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | - | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### samba-forward / bytes (neural shape small: B2 L64 DM64 V256 H2 FF128 layers mamba3+attention)

race: failed, driver rc 3, log `logs/neural.samba-forward.bytes.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### samba-infer / bytes (neural shape small: B2 L64 DM64 V256 H2 FF128 layers mamba3+attention)

race: failed, driver rc 3, log `logs/neural.samba-infer.bytes.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### samba-train-step / bytes (neural shape small: B2 L64 DM64 V256 H2 FF128 layers mamba3+attention)

race: failed, driver rc 3, log `logs/neural.samba-train-step.bytes.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| amsgrad | false | false | false | false | false |
| betas | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] | [0.9, 0.999] |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eps | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| seed | - | 7 | 7 | 7 | 7 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### transformer-forward / gaussian (neural shape small: B2 L64 DM64 H4 KV2 HD16 FF128)

race: failed, driver rc 3, log `logs/neural.transformer-forward.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-fp32 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-eager-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-compile-bf16 | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-compile-bf16 | torch-compile-fp32 | torch-eager-bf16 | torch-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### transformer-infer / gaussian (neural shape small: B2 L64 DM64 H4 KV2 HD16 FF128)

race: failed, driver rc 3, log `logs/neural.transformer-infer.gaussian.shape-small.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | cpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-fp32 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-eager-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |
| torch-cpu-compile-bf16 | torch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none") |

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | torch-cpu-compile-bf16 | torch-cpu-compile-fp32 | torch-cpu-eager-bf16 | torch-cpu-eager-fp32 |
|---|---||---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | torch (declared) | torch (declared) | torch (declared) | torch (declared) |
| eps | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| seed | - | 7 | 7 | 7 | 7 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

## Algorithm expansion

### additive-chi2 / istella (rows 2000, shape X 2000x220; Xq 1000x220; y 2000; yq 1000)

race: failed, driver rc 3, log `logs/algos.additive-chi2.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'sample_steps': 2}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### additive-chi2 / taxi (rows 2000, shape X 2000x11; Xq 1000x11; y 2000; yq 1000)

race: failed, driver rc 3, log `logs/algos.additive-chi2.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'sample_steps': 2}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### affinity-prop / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.affinity-prop.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 204.1 | 204.1..204.1 | 1 | - | - | - | 564.6 | - | n_clusters=184, silhouette=0.090437 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 195.2 | 195.2..195.2 | 1 | - | - | - | 565.4 | - | ari_vs_ours=1.000000, n_clusters=184, silhouette=0.090437 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 550.5 | 550.5..550.5 | 1 | 0.371 | 0.355 | - | 345.9 | - | ari_vs_ours=1.000000, n_clusters=184, silhouette=0.090437 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'affinity': 'euclidean', 'convergence_iter': 15, 'damping': 0.5, 'max_iter': 200, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| affinity | "euclidean" | "euclidean" | "euclidean" |
| max_iter | 200 | 200 | 200 |
| seed | 7 | 7 | 7 |

### affinity-prop / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.affinity-prop.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 232.7 | 232.7..232.7 | 1 | - | - | - | 551.8 | - | n_clusters=148, silhouette=0.177010 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 227.7 | 227.7..227.7 | 1 | - | - | - | 552.3 | - | ari_vs_ours=1.000000, n_clusters=148, silhouette=0.177010 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 816.7 | 816.7..816.7 | 1 | 0.285 | 0.279 | - | 341.5 | - | ari_vs_ours=1.000000, n_clusters=148, silhouette=0.177010 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'affinity': 'euclidean', 'convergence_iter': 15, 'damping': 0.5, 'max_iter': 200, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| affinity | "euclidean" | "euclidean" | "euclidean" |
| max_iter | 200 | 200 | 200 |
| seed | 7 | 7 | 7 |

### ard / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.ard.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### ard / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.ard.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'threshold_lambda': 10000.0, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### bayesian-gmm / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 1, log `logs/algos.bayesian-gmm.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) |

settings: {'covariance_type': 'full', 'init_params': 'kmeans', 'max_iter': 100, 'n_components': 8, 'n_init': 1, 'random_state': 7, 'reg_covar': 1e-06, 'tol': 0.001, 'weight_concentration_prior_type': 'dirichlet_process'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### bayesian-gmm / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.bayesian-gmm.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 185.1 | 185.1..185.1 | 1 | - | - | - | 337.9 | - | mean_log_likelihood=-3.296631 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 215.2 | 215.2..215.2 | 1 | - | - | - | 335.5 | - | mean_log_likelihood=-3.296631 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 48.2 | 48.2..48.2 | 1 | 3.842 | 4.468 | - | 152.0 | - | mean_log_likelihood=-1.672639 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'covariance_type': 'full', 'init_params': 'kmeans', 'max_iter': 100, 'n_components': 8, 'n_init': 1, 'random_state': 7, 'reg_covar': 1e-06, 'tol': 0.001, 'weight_concentration_prior_type': 'dirichlet_process'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### bayesian-ridge / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.bayesian-ridge.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### bayesian-ridge / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.bayesian-ridge.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'max_iter': 300, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| max_iter | 300 | 300 | 300 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### bisecting-kmeans / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.bisecting-kmeans.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 98.5 | 98.5..98.5 | 1 | - | - | - | 350.4 | - | n_clusters=8, silhouette=0.127796 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 90.7 | 90.7..90.7 | 1 | - | - | - | 349.3 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.127796 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 63.5 | 63.5..63.5 | 1 | 1.551 | 1.428 | - | 181.4 | - | ari_vs_ours=0.840882, n_clusters=8, silhouette=0.121102 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'lloyd', 'bisecting_strategy': 'biggest_inertia', 'init': 'random', 'max_iter': 300, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### bisecting-kmeans / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.bisecting-kmeans.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 66.1 | 66.1..66.1 | 1 | - | - | - | 330.2 | - | n_clusters=8, silhouette=0.153386 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 68.1 | 68.1..68.1 | 1 | - | - | - | 330.2 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.153386 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 33.6 | 33.6..33.6 | 1 | 1.966 | 2.024 | - | 146.9 | - | ari_vs_ours=0.540882, n_clusters=8, silhouette=0.146108 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'lloyd', 'bisecting_strategy': 'biggest_inertia', 'init': 'random', 'max_iter': 300, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### cca / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.cca.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### cca / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.cca.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### classical-mds / istella (rows 2000, shape X 2000x220; Xq 200x220)

race: failed, driver rc 3, log `logs/algos.classical-mds.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'n_components': 2}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | __main__ (attributes) |
| metric | "euclidean" | "euclidean" | - |
| n_components | 2 | 2 | 2 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (__main__): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### classical-mds / taxi (rows 2000, shape X 2000x11; Xq 200x11)

race: failed, driver rc 3, log `logs/algos.classical-mds.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'n_components': 2}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | __main__ (attributes) |
| metric | "euclidean" | "euclidean" | - |
| n_components | 2 | 2 | 2 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (__main__): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### connected-components / istella (rows 2000, shape X 2000x220; indices 30756; indices2 6214; indptr 2001; indptr2 2001; y 2000)

race: failed, driver rc 3, log `logs/algos.connected-components.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| networkx-cpu | networkx | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: networkx-cpu (networkx): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### connected-components / taxi (rows 2000, shape X 2000x11; indices 26186; indices2 5530; indptr 2001; indptr2 2001; y 2000)

race: failed, driver rc 3, log `logs/algos.connected-components.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| networkx-cpu | networkx | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: networkx-cpu (networkx): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### dict-learning / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.dict-learning.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8020.3 | 8020.3..8020.3 | 1 | - | - | - | 396.0 | - | component_sparsity=0.104545, relative_reconstruction_error=0.529917 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7490.9 | 7490.9..7490.9 | 1 | - | - | - | 381.7 | - | component_sparsity=0.104545, relative_reconstruction_error=0.529917 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1663.1 | 1663.1..1663.1 | 1 | 4.823 | 4.504 | - | 165.3 | - | component_sparsity=0.104545, relative_reconstruction_error=0.529918 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_algorithm': 'cd', 'max_iter': 100, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'split_sign': False, 'tol': 1e-08, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |
| tol | 1e-08 | 1e-08 | 1e-08 |

### dict-learning / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.dict-learning.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4888.5 | 4888.5..4888.5 | 1 | - | - | - | 361.8 | - | component_sparsity=0.000000, relative_reconstruction_error=0.471369 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4726.9 | 4726.9..4726.9 | 1 | - | - | - | 357.7 | - | component_sparsity=0.000000, relative_reconstruction_error=0.471369 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1553.7 | 1553.7..1553.7 | 1 | 3.146 | 3.042 | - | 148.0 | - | component_sparsity=0.000000, relative_reconstruction_error=0.467216 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_algorithm': 'cd', 'max_iter': 100, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'split_sign': False, 'tol': 1e-08, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |
| tol | 1e-08 | 1e-08 | 1e-08 |

### elliptic-envelope / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.elliptic-envelope.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 219026.4 | 219026.4..219026.4 | 1 | - | - | - | 422.5 | - | fraction_flagged=0.100500, jaccard_vs_sklearn=0.139130 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 39430.0 | 39430.0..39430.0 | 1 | - | - | - | 419.2 | - | fraction_flagged=0.100500, jaccard_vs_sklearn=0.139130 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2712.7 | 2712.7..2712.7 | 1 | 80.742 | 14.535 | - | 255.6 | - | fraction_flagged=0.096000, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'contamination': 0.1, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| contamination | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 |

### elliptic-envelope / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.elliptic-envelope.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 319.2 | 319.2..319.2 | 1 | - | - | - | 358.9 | - | fraction_flagged=0.101500, jaccard_vs_sklearn=0.750000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 251.0 | 251.0..251.0 | 1 | - | - | - | 358.7 | - | fraction_flagged=0.101500, jaccard_vs_sklearn=0.750000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 94.8 | 94.8..94.8 | 1 | 3.366 | 2.647 | - | 147.0 | - | fraction_flagged=0.091000, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'contamination': 0.1, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| contamination | 0.1 | 0.1 | 0.1 |
| seed | 7 | 7 | 7 |

### enet-cv / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.enet-cv.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 28362.1 | 28362.1..28362.1 | 1 | - | - | - | 519.5 | - | finite=True, r2=0.275634, rmse=0.675633 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 26880.3 | 26880.3..26880.3 | 1 | - | - | - | 521.8 | - | finite=True, r2=0.275639, rmse=0.675630 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 75.0 | 75.0..75.0 | 1 | 378.270 | 358.507 | - | 162.6 | - | finite=True, r2=0.275634, rmse=0.675632 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| l1_ratio | 0.5 | 0.5 | 0.5 |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### enet-cv / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.enet-cv.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 19.6 | 19.6..19.6 | 1 | - | - | - | 471.1 | - | finite=True, r2=0.930720, rmse=3.844766 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13.4 | 13.4..13.4 | 1 | - | - | - | 467.0 | - | finite=True, r2=0.930720, rmse=3.844767 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 93.9 | 93.9..93.9 | 1 | 0.209 | 0.142 | - | 146.5 | - | finite=True, r2=0.930720, rmse=3.844767 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| l1_ratio | 0.5 | 0.5 | 0.5 |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### factor-analysis / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.factor-analysis.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 22071.3 | 22071.3..22071.3 | 1 | - | - | - | 392.9 | - | mean_log_likelihood=-2.5e+08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 18046.7 | 18046.7..18046.7 | 1 | - | - | - | 378.2 | - | mean_log_likelihood=-2.5e+08 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 130.2 | 130.2..130.2 | 1 | 169.543 | 138.627 | - | 148.8 | - | mean_log_likelihood=-2.5e+08 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'iterated_power': 3, 'max_iter': 1000, 'n_components': 8, 'random_state': 7, 'svd_method': 'randomized', 'tol': 0.01}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 1000 | 1000 | 1000 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0.01 | 0.01 | 0.01 |

### factor-analysis / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.factor-analysis.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 117.8 | 117.8..117.8 | 1 | - | - | - | 357.7 | - | mean_log_likelihood=-15.230434 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 108.6 | 108.6..108.6 | 1 | - | - | - | 356.6 | - | mean_log_likelihood=-15.230434 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34.0 | 34.0..34.0 | 1 | 3.465 | 3.194 | - | 145.0 | - | mean_log_likelihood=-15.230489 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'iterated_power': 3, 'max_iter': 1000, 'n_components': 8, 'random_state': 7, 'svd_method': 'randomized', 'tol': 0.01}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 1000 | 1000 | 1000 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0.01 | 0.01 | 0.01 |

### fastica / istella (rows 2000, shape X 1800x220; Xq 200x220)

race: done, driver rc 0, log `logs/algos.fastica.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 197.7 | 197.7..197.7 | 1 | - | - | - | 368.7 | - | mean_abs_excess_kurtosis=80.895565 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 126.1 | 126.1..126.1 | 1 | - | - | - | 365.3 | - | mean_abs_excess_kurtosis=80.899842 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14.4 | 14.4..14.4 | 1 | 13.718 | 8.749 | - | 153.3 | - | mean_abs_excess_kurtosis=83.835013 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'parallel', 'fun': 'logcosh', 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'tol': 0.0001, 'whiten': 'unit-variance', 'whiten_solver': 'svd'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "parallel" | "parallel" | "parallel" |
| max_iter | 200 | 200 | 200 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |
| whiten | "unit-variance" | "unit-variance" | "unit-variance" |

### fastica / taxi (rows 2000, shape X 1800x11; Xq 200x11)

race: done, driver rc 0, log `logs/algos.fastica.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 51.2 | 51.2..51.2 | 1 | - | - | - | 358.1 | - | mean_abs_excess_kurtosis=4.956103 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46.5 | 46.5..46.5 | 1 | - | - | - | 357.7 | - | mean_abs_excess_kurtosis=4.956102 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.7 | 2.7..2.7 | 1 | 18.802 | 17.067 | - | 141.4 | - | mean_abs_excess_kurtosis=4.948144 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'algorithm': 'parallel', 'fun': 'logcosh', 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'tol': 0.0001, 'whiten': 'unit-variance', 'whiten_solver': 'svd'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "parallel" | "parallel" | "parallel" |
| max_iter | 200 | 200 | 200 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0.0001 | 0.0001 | 0.0001 |
| whiten | "unit-variance" | "unit-variance" | "unit-variance" |

### gamma / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.gamma.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### gamma / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.gamma.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### gaussian-rp / istella (rows 2000, shape X 1800x220; Xq 200x220)

race: done, driver rc 0, log `logs/algos.gaussian-rp.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.5 | 3.5..3.5 | 1 | - | - | - | 331.8 | - | mean_abs_distortion=0.136908 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2.9 | 2.9..2.9 | 1 | - | - | - | 329.8 | - | mean_abs_distortion=0.136908 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.8 | 0.8..0.8 | 1 | 4.639 | 3.800 | - | 145.1 | - | mean_abs_distortion=0.232702 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 'half', 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 110 | 110 | 110 |
| seed | 7 | 7 | 7 |

### gaussian-rp / taxi (rows 2000, shape X 1800x11; Xq 200x11)

race: done, driver rc 0, log `logs/algos.gaussian-rp.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.8 | 2.8..2.8 | 1 | - | - | - | 321.3 | - | mean_abs_distortion=0.418718 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2.6 | 2.6..2.6 | 1 | - | - | - | 319.1 | - | mean_abs_distortion=0.418718 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.4 | 0.4..0.4 | 1 | 7.362 | 6.759 | - | 142.7 | - | mean_abs_distortion=0.455943 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'eps': 0.1, 'n_components': 'half', 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 5 | 5 | 5 |
| seed | 7 | 7 | 7 |

### huber / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.huber.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'epsilon': 1.35, 'fit_intercept': True, 'max_iter': 100, 'tol': 1e-05}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 1.35 | 1.35 | 1.35 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| tol | 1e-05 | 1e-05 | 1e-05 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### huber / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.huber.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'epsilon': 1.35, 'fit_intercept': True, 'max_iter': 100, 'tol': 1e-05}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 1.35 | 1.35 | 1.35 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| tol | 1e-05 | 1e-05 | 1e-05 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### incremental-pca / istella (rows 2000, shape X 1800x220; Xq 200x220)

race: failed, driver rc 3, log `logs/algos.incremental-pca.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'batch_size': 65536, 'n_components': 8, 'whiten': False}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 65536 | 65536 | 65536 |
| n_components | 8 | 8 | 8 |
| whiten | false | false | false |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### incremental-pca / taxi (rows 2000, shape X 1800x11; Xq 200x11)

race: failed, driver rc 3, log `logs/algos.incremental-pca.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'batch_size': 65536, 'n_components': 8, 'whiten': False}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 65536 | 65536 | 65536 |
| n_components | 8 | 8 | 8 |
| whiten | false | false | false |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### isomap / istella (rows 2000, shape X 2000x220; Xq 200x220)

race: failed, driver rc 3, log `logs/algos.isomap.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'eigen_solver': 'auto', 'metric': 'minkowski', 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'p': 2, 'path_method': 'auto', 'tol': 0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | null |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| tol | - | - | 0 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### isomap / taxi (rows 2000, shape X 2000x11; Xq 200x11)

race: failed, driver rc 3, log `logs/algos.isomap.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'eigen_solver': 'auto', 'metric': 'minkowski', 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'p': 2, 'path_method': 'auto', 'tol': 0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | null |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| p | 2 | 2 | 2 |
| tol | - | - | 0 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### isotonic / istella (rows 2000, shape X 2000; Xq 2000; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.isotonic.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'increasing': True, 'out_of_bounds': 'clip'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### isotonic / taxi (rows 2000, shape X 2000; Xq 2000; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.isotonic.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'increasing': True, 'out_of_bounds': 'clip'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### kernel-pca / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.kernel-pca.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 99361.4 | 99361.4..99361.4 | 1 | - | - | - | 439.4 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 56803.8 | 56803.8..56803.8 | 1 | - | - | - | 439.2 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 28.4 | 28.4..28.4 | 1 | 3501.622 | 2001.837 | - | 255.8 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| coef0 | 1 | 1 | 1 |
| degree | 3 | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | null | null | null |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0 | 0 | 0 |

accepted difference: ours-fast max_iter: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu max_iter: None on both ours and sklearn-cpu: the same documented setting in both signatures

### kernel-pca / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.kernel-pca.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 136095.4 | 136095.4..136095.4 | 1 | - | - | - | 431.9 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 77444.9 | 77444.9..77444.9 | 1 | - | - | - | 431.9 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 24.0 | 24.0..24.0 | 1 | 5668.575 | 3225.694 | - | 247.0 | - | subspace_cos_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'coef0': 1, 'degree': 3, 'eigen_solver': 'auto', 'fit_inverse_transform': False, 'iterated_power': 'auto', 'kernel': 'rbf', 'n_components': 8, 'random_state': 7, 'remove_zero_eig': False, 'tol': 0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| coef0 | 1 | 1 | 1 |
| degree | 3 | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | null | null | null |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 0 | 0 | 0 |

accepted difference: ours-fast max_iter: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu max_iter: None on both ours and sklearn-cpu: the same documented setting in both signatures

### knn-imputer / istella (rows 2000, shape X 2000x220; X_true 2000x220; Xq 2000x220; Xq_true 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.knn-imputer.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| metric | "nan_euclidean" | "nan_euclidean" | "nan_euclidean" |
| n_neighbors | 5 | 5 | 5 |
| weights | "uniform" | "uniform" | "uniform" |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### knn-imputer / taxi (rows 2000, shape X 2000x11; X_true 2000x11; Xq 2000x11; Xq_true 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.knn-imputer.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'add_indicator': False, 'keep_empty_features': False, 'metric': 'nan_euclidean', 'n_neighbors': 5, 'weights': 'uniform'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| metric | "nan_euclidean" | "nan_euclidean" | "nan_euclidean" |
| n_neighbors | 5 | 5 | 5 |
| weights | "uniform" | "uniform" | "uniform" |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### label-propagation / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; y_semi 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.label-propagation.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'gamma': 20, 'kernel': 'knn', 'max_iter': 1000, 'n_neighbors': 7, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 1000 | 1000 | 1000 |
| n_neighbors | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### label-propagation / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; y_semi 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.label-propagation.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'gamma': 20, 'kernel': 'knn', 'max_iter': 1000, 'n_neighbors': 7, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 1000 | 1000 | 1000 |
| n_neighbors | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### label-spreading / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; y_semi 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.label-spreading.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.2, 'gamma': 20, 'kernel': 'knn', 'max_iter': 30, 'n_neighbors': 7, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.2 | 0.2 | 0.2 |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 30 | 30 | 30 |
| n_neighbors | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### label-spreading / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; y_semi 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.label-spreading.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.2, 'gamma': 20, 'kernel': 'knn', 'max_iter': 30, 'n_neighbors': 7, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.2 | 0.2 | 0.2 |
| gamma | 20 | 20 | 20 |
| kernel | "knn" | "knn" | "knn" |
| max_iter | 30 | 30 | 30 |
| n_neighbors | 7 | 7 | 7 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### lars / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.lars.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 18235.2 | 18235.2..18235.2 | 1 | - | - | - | 522.8 | - | finite=True, r2=-1.609e+27, rmse=3.185e+13 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 17498.6 | 17498.6..17498.6 | 1 | - | - | - | 506.3 | - | finite=True, r2=-7.339e+30, rmse=2.151e+15 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14.1 | 14.1..14.1 | 1 | 1293.462 | 1241.209 | - | 152.1 | - | finite=True, r2=-2.19e+49, rmse=3.715e+24 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |

### lars / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.lars.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.3 | 3.3..3.3 | 1 | - | - | - | 470.5 | - | finite=True, r2=0.930140, rmse=3.860828 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2.9 | 2.9..2.9 | 1 | - | - | - | 467.5 | - | finite=True, r2=0.930140, rmse=3.860829 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.9 | 0.9..0.9 | 1 | 3.660 | 3.255 | - | 143.8 | - | finite=True, r2=0.930140, rmse=3.860823 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eps': 2.220446049250313e-16, 'fit_intercept': True, 'n_nonzero_coefs': 500, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 2.220446049250313e-16 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| fit_intercept | true | true | true |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |

### lasso-cv / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.lasso-cv.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20290.6 | 20290.6..20290.6 | 1 | - | - | - | 540.7 | - | finite=True, r2=0.276319, rmse=0.675313 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 19181.7 | 19181.7..19181.7 | 1 | - | - | - | 505.7 | - | finite=True, r2=0.276319, rmse=0.675313 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 56.8 | 56.8..56.8 | 1 | 357.432 | 337.897 | - | 161.0 | - | finite=True, r2=0.276321, rmse=0.675312 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### lasso-cv / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.lasso-cv.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 13.5 | 13.5..13.5 | 1 | - | - | - | 469.2 | - | finite=True, r2=0.930795, rmse=3.842696 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 11.2 | 11.2..11.2 | 1 | - | - | - | 467.3 | - | finite=True, r2=0.930795, rmse=3.842696 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 93.9 | 93.9..93.9 | 1 | 0.143 | 0.120 | - | 146.2 | - | finite=True, r2=0.930795, rmse=3.842697 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alphas': [0.001, 0.01, 0.1, 1.0], 'cv': 5, 'eps': 0.001, 'fit_intercept': True, 'max_iter': 1000, 'positive': False, 'random_state': 7, 'selection': 'cyclic', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.001 | 0.001 | 0.001 |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| positive | false | false | false |
| precompute | "auto" | "auto" | "auto" |
| seed | 7 | 7 | 7 |
| selection | "cyclic" | "cyclic" | "cyclic" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### lasso-lars / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.lasso-lars.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1179.3 | 1179.3..1179.3 | 1 | - | - | - | 478.1 | - | finite=True, r2=0.263997, rmse=0.681038 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 691.5 | 691.5..691.5 | 1 | - | - | - | 475.3 | - | finite=True, r2=0.276169, rmse=0.675383 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 540.297 | 316.809 | - | 152.1 | - | finite=True, r2=0.256262, rmse=0.684607 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### lasso-lars / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.lasso-lars.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.4 | 3.4..3.4 | 1 | - | - | - | 466.2 | - | finite=True, r2=0.930257, rmse=3.857584 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.1 | 3.1..3.1 | 1 | - | - | - | 465.9 | - | finite=True, r2=0.930257, rmse=3.857584 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.9 | 0.9..0.9 | 1 | 3.849 | 3.490 | - | 143.9 | - | finite=True, r2=0.930258, rmse=3.857579 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.01, 'eps': 2.220446049250313e-16, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### lda / taxi-zones (rows 2000, shape X 1162x241; Xq 130x241)

race: done, driver rc 0, log `logs/algos.lda.taxi-zones.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4457.1 | 4457.1..4457.1 | 1 | - | - | - | 383.2 | - | perplexity=54.130923 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4108.3 | 4108.3..4108.3 | 1 | - | - | - | 384.6 | - | perplexity=54.130916 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1072.8 | 1072.8..1072.8 | 1 | 4.155 | 3.829 | - | 159.4 | - | perplexity=54.602803 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 128, 'learning_decay': 0.7, 'learning_method': 'batch', 'learning_offset': 10.0, 'max_doc_update_iter': 100, 'max_iter': 20, 'mean_change_tol': 0.001, 'n_components': 16, 'perp_tol': 0.1, 'random_state': 7, 'total_samples': 1000000.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 128 | 128 | 128 |
| max_iter | 20 | 20 | 20 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |

### lda / text (rows 2000, shape X 3600x4096; Xq 400x4096; y 3600; yq 400)

race: done, driver rc 0, log `logs/algos.lda.text.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 222874.4 | 222874.4..222874.4 | 1 | - | - | - | 1901.6 | - | perplexity=267.071498 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 211016.3 | 211016.3..211016.3 | 1 | - | - | - | 1901.3 | - | perplexity=267.071491 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4934.7 | 4934.7..4934.7 | 1 | 45.165 | 42.762 | - | 242.1 | - | perplexity=266.677528 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 128, 'learning_decay': 0.7, 'learning_method': 'batch', 'learning_offset': 10.0, 'max_doc_update_iter': 100, 'max_iter': 20, 'mean_change_tol': 0.001, 'n_components': 16, 'perp_tol': 0.1, 'random_state': 7, 'total_samples': 1000000.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| batch_size | 128 | 128 | 128 |
| max_iter | 20 | 20 | 20 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |

### lle / istella (rows 2000, shape X 2000x220; Xq 200x220)

race: done, driver rc 0, log `logs/algos.lle.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | 533820.8 | 533820.8..533820.8 | 1 | - | - | - | 559.6 | - | trustworthiness_k15=0.743021 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 155.3 | 155.3..155.3 | 1 | - | 3436.337 | - | 194.2 | - | trustworthiness_k15=0.742862 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours: host not sampled; GPU not sampled

memory, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eigen_solver': 'auto', 'hessian_tol': 0.0001, 'max_iter': 100, 'method': 'standard', 'modified_tol': 1e-12, 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'random_state': 7, 'reg': 0.001, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | 100 |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| tol | - | - | 1e-06 |

### lle / taxi (rows 2000, shape X 2000x11; Xq 200x11)

race: done, driver rc 0, log `logs/algos.lle.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(timeout: null) |
| sklearn-cpu | scikit-learn | cpu | opponent | 153.9 | 153.9..153.9 | 1 | - | - | - | 168.8 | - | trustworthiness_k15=0.719153 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host not sampled; GPU not sampled

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'eigen_solver': 'auto', 'hessian_tol': 0.0001, 'max_iter': 100, 'method': 'standard', 'modified_tol': 1e-12, 'n_components': 2, 'n_neighbors': 10, 'neighbors_algorithm': 'auto', 'random_state': 7, 'reg': 0.001, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | - | - | 100 |
| n_components | 2 | 2 | 2 |
| n_neighbors | 10 | 10 | 10 |
| seed | 7 | 7 | 7 |
| tol | - | - | 1e-06 |

### lof / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.lof.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'algorithm': 'brute', 'contamination': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'n_neighbors': 20, 'novelty': False, 'p': 2}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| contamination | "auto" | "auto" | "auto" |
| leaf_size | 30 | 30 | 30 |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_neighbors | 20 | 20 | 20 |
| p | 2 | 2 | 2 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### lof / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.lof.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'algorithm': 'brute', 'contamination': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'n_neighbors': 20, 'novelty': False, 'p': 2}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "brute" | "brute" | "brute" |
| contamination | "auto" | "auto" | "auto" |
| leaf_size | 30 | 30 | 30 |
| metric | "minkowski" | "minkowski" | "minkowski" |
| n_neighbors | 20 | 20 | 20 |
| p | 2 | 2 | 2 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### logreg-cv / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.logreg-cv.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3509.0 | 3509.0..3509.0 | 1 | - | - | - | 473.5 | - | accuracy=0.922000, logloss=0.192201 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3222.1 | 3222.1..3222.1 | 1 | - | - | - | 485.8 | - | accuracy=0.922000, logloss=0.192123 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1104.5 | 1104.5..1104.5 | 1 | 3.177 | 2.917 | - | 156.9 | - | accuracy=0.922000, logloss=0.192195 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'Cs': [0.1, 1.0, 10.0], 'cv': 5, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1.0, 'max_iter': 1000, 'penalty': 'l2', 'random_state': 7, 'refit': True, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

### logreg-cv / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.logreg-cv.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 42.0 | 42.0..42.0 | 1 | - | - | - | 467.1 | - | accuracy=0.763000, logloss=0.547045 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 33.2 | 33.2..33.2 | 1 | - | - | - | 468.3 | - | accuracy=0.763000, logloss=0.547045 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 904.6 | 904.6..904.6 | 1 | 0.046 | 0.037 | - | 149.9 | - | accuracy=0.763000, logloss=0.547041 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'Cs': [0.1, 1.0, 10.0], 'cv': 5, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1.0, 'max_iter': 1000, 'penalty': 'l2', 'random_state': 7, 'refit': True, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| class_weight | null | null | null |
| fit_intercept | true | true | true |
| max_iter | 1000 | 1000 | 1000 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

### louvain / istella (rows 2000, shape X 2000x220; indices 30756; indices2 6214; indptr 2001; indptr2 2001; y 2000)

race: done, driver rc 0, log `logs/algos.louvain.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 466.8 | 466.8..466.8 | 1 | - | - | - | 280.4 | - | modularity=0.813547, n_communities=14 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 233.5 | 233.5..233.5 | 1 | - | - | - | 283.6 | - | modularity=0.813547, n_communities=14 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 82.3 | 82.3..82.3 | 1 | 5.671 | 2.837 | - | 85.7 | - | modularity=0.813989, n_communities=14 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | 7 |

### louvain / taxi (rows 2000, shape X 2000x11; indices 26186; indices2 5530; indptr 2001; indptr2 2001; y 2000)

race: done, driver rc 0, log `logs/algos.louvain.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 466.1 | 466.1..466.1 | 1 | - | - | - | 275.0 | - | modularity=0.795766, n_communities=19 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 255.1 | 255.1..255.1 | 1 | - | - | - | 277.4 | - | modularity=0.795766, n_communities=19 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| networkx-cpu | networkx | cpu | opponent | 73.4 | 73.4..73.4 | 1 | 6.346 | 3.474 | - | 81.0 | - | modularity=0.800402, n_communities=17 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, networkx-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'resolution': 1.0, 'seed': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are order-dependent; ours pins the vertex sweep (lowest id first)

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | 7 |

### lstsq / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.lstsq.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| numpy-cpu | numpy | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: numpy-cpu (numpy): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: torch-gpu (torch): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### lstsq / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.lstsq.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| numpy-cpu | numpy | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: numpy-cpu (numpy): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: torch-gpu (torch): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### lu-solve / synthetic (rows 2000, shape -)

race: failed, driver rc 3, log `logs/algos.lu-solve.synthetic.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| numpy-cpu | numpy | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (4): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | numpy-cpu | ours | ours-fast | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | numpy (declared) | mojolearn (declared) | mojolearn (declared) | torch (declared) |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: numpy-cpu (numpy): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: torch-gpu (torch): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### mb-dict-learning / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.mb-dict-learning.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6752.3 | 6752.3..6752.3 | 1 | - | - | - | 394.6 | - | component_sparsity=0.104545, relative_reconstruction_error=0.537705 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6364.7 | 6364.7..6364.7 | 1 | - | - | - | 382.1 | - | component_sparsity=0.104545, relative_reconstruction_error=0.537705 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4071.9 | 4071.9..4071.9 | 1 | 1.658 | 1.563 | - | 185.6 | - | component_sparsity=0.104545, relative_reconstruction_error=0.541634 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 256, 'fit_algorithm': 'lars', 'max_iter': 10, 'max_no_improvement': 10, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'shuffle': True, 'split_sign': False, 'tol': 0.001, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| batch_size | 256 | 256 | 256 |
| max_iter | 10 | 10 | 10 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.001 | 0.001 | 0.001 |

### mb-dict-learning / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.mb-dict-learning.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2864.4 | 2864.4..2864.4 | 1 | - | - | - | 360.4 | - | component_sparsity=0.000000, relative_reconstruction_error=0.478903 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2747.0 | 2747.0..2747.0 | 1 | - | - | - | 358.2 | - | component_sparsity=0.000000, relative_reconstruction_error=0.478903 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3810.4 | 3810.4..3810.4 | 1 | 0.752 | 0.721 | - | 152.0 | - | component_sparsity=0.000000, relative_reconstruction_error=0.468998 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 256, 'fit_algorithm': 'lars', 'max_iter': 10, 'max_no_improvement': 10, 'n_components': 16, 'positive_code': False, 'positive_dict': False, 'random_state': 7, 'shuffle': True, 'split_sign': False, 'tol': 0.001, 'transform_algorithm': 'lasso_cd', 'transform_max_iter': 1000}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| batch_size | 256 | 256 | 256 |
| max_iter | 10 | 10 | 10 |
| n_components | 16 | 16 | 16 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.001 | 0.001 | 0.001 |

### mb-sparse-pca / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.mb-sparse-pca.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2065.7 | 2065.7..2065.7 | 1 | - | - | - | 400.4 | - | component_sparsity=0.355114, relative_reconstruction_error=0.675542 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1782.7 | 1782.7..1782.7 | 1 | - | - | - | 407.6 | - | component_sparsity=0.355114, relative_reconstruction_error=0.675542 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 518.0 | 518.0..518.0 | 1 | 3.988 | 3.441 | - | 166.8 | - | component_sparsity=0.355114, relative_reconstruction_error=0.675542 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 1024, 'max_iter': 10, 'max_no_improvement': 10, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'shuffle': True, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| batch_size | 1024 | 1024 | 1024 |
| max_iter | 10 | 10 | 10 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.001 | 0.001 | 0.001 |

### mb-sparse-pca / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.mb-sparse-pca.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 236.0 | 236.0..236.0 | 1 | - | - | - | 361.7 | - | component_sparsity=0.250000, relative_reconstruction_error=0.279922 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 213.3 | 213.3..213.3 | 1 | - | - | - | 358.1 | - | component_sparsity=0.250000, relative_reconstruction_error=0.279922 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2253.5 | 2253.5..2253.5 | 1 | 0.105 | 0.095 | - | 155.4 | - | component_sparsity=0.250000, relative_reconstruction_error=0.279922 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'batch_size': 1024, 'max_iter': 10, 'max_no_improvement': 10, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'shuffle': True, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| batch_size | 1024 | 1024 | 1024 |
| max_iter | 10 | 10 | 10 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | 0.001 | 0.001 | 0.001 |

### mds / istella (rows 2000, shape X 2000x220; Xq 200x220)

race: failed, driver rc 3, log `logs/algos.mds.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu") |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu") |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu") |

settings: {'eps': 0.001, 'max_iter': 300, 'n_components': 2, 'n_init': 1, 'normalized_stress': 'auto', 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: metric MDS on Euclidean distances on both, spelled differently: ours metric='euclidean', metric_mds=True, init='random' (scikit-learn 1.9's names); the pinned scikit-learn 1.7.2 metric=True, dissimilarity='euclidean' and a random start from random_state; each draws its own start

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.001 | 0.001 | 0.001 |
| init | "random" | "random" | - |
| max_iter | 300 | 300 | 300 |
| metric | "euclidean" | "euclidean" | true |
| n_components | 2 | 2 | 2 |
| n_init | 1 | 1 | 1 |
| seed | 7 | 7 | 7 |

REFUSED: sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu

### mds / taxi (rows 2000, shape X 2000x11; Xq 200x11)

race: failed, driver rc 3, log `logs/algos.mds.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu") |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu") |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (1): sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu") |

settings: {'eps': 0.001, 'max_iter': 300, 'n_components': 2, 'n_init': 1, 'normalized_stress': 'auto', 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: metric MDS on Euclidean distances on both, spelled differently: ours metric='euclidean', metric_mds=True, init='random' (scikit-learn 1.9's names); the pinned scikit-learn 1.7.2 metric=True, dissimilarity='euclidean' and a random start from random_state; each draws its own start

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.001 | 0.001 | 0.001 |
| init | "random" | "random" | - |
| max_iter | 300 | 300 | 300 |
| metric | "euclidean" | "euclidean" | true |
| n_components | 2 | 2 | 2 |
| n_init | 1 | 1 | 1 |
| seed | 7 | 7 | 7 |

REFUSED: sklearn-cpu: metric is 'euclidean' (metric) on ours and True (metric) on sklearn-cpu

### meanshift / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.meanshift.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'bin_seeding': True, 'cluster_all': True, 'max_iter': 300, 'min_bin_freq': 1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bandwidth | 12.656069834118984 | 12.656069834118984 | 12.656069834118984 |
| max_iter | 300 | 300 | 300 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### meanshift / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.meanshift.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'bin_seeding': True, 'cluster_all': True, 'max_iter': 300, 'min_bin_freq': 1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| bandwidth | 2.633764493961558 | 2.633764493961558 | 2.633764493961558 |
| max_iter | 300 | 300 | 300 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### min-cov-det / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.min-cov-det.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 302.9 | 302.9..302.9 | 1 | - | - | - | 358.3 | - | n_features=11, rel_diff_vs_sklearn=0.133906 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 229.1 | 229.1..229.1 | 1 | - | - | - | 357.7 | - | n_features=11, rel_diff_vs_sklearn=0.133907 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 95.5 | 95.5..95.5 | 1 | 3.173 | 2.400 | - | 147.7 | - | n_features=11 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| seed | 7 | 7 | 7 |

### minibatch-kmeans / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.minibatch-kmeans.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 71.0 | 71.0..71.0 | 1 | - | - | - | 341.3 | - | n_clusters=8, silhouette=0.104964 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 71.0 | 71.0..71.0 | 1 | - | - | - | 341.6 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.104964 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 57.4 | 57.4..57.4 | 1 | 1.238 | 1.238 | - | 191.7 | - | ari_vs_ours=0.529562, n_clusters=8, silhouette=0.095856 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### minibatch-kmeans / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.minibatch-kmeans.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 50.8 | 50.8..50.8 | 1 | - | - | - | 321.8 | - | n_clusters=8, silhouette=0.166913 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 49.9 | 49.9..49.9 | 1 | - | - | - | 321.4 | - | ari_vs_ours=1.000000, n_clusters=8, silhouette=0.166913 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 63.3 | 63.3..63.3 | 1 | 0.802 | 0.789 | - | 149.7 | - | ari_vs_ours=0.405726, n_clusters=8, silhouette=0.143149 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'batch_size': 4096, 'init': 'k-means++', 'max_iter': 100, 'max_no_improvement': 10, 'n_clusters': 8, 'n_init': 1, 'random_state': 7, 'reassignment_ratio': 0.01, 'tol': 0.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### nearest-centroid / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.nearest-centroid.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| metric | "euclidean" | "euclidean" | "euclidean" |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### nearest-centroid / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.nearest-centroid.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'metric': 'euclidean', 'priors': 'uniform'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| metric | "euclidean" | "euclidean" | "euclidean" |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### nmf / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.nmf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2154.6 | 2154.6..2154.6 | 1 | - | - | - | 403.9 | - | relative_reconstruction_error=0.332851 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1862.6 | 1862.6..1862.6 | 1 | - | - | - | 403.5 | - | relative_reconstruction_error=0.332851 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 34.5 | 34.5..34.5 | 1 | 62.367 | 53.914 | - | 150.5 | - | relative_reconstruction_error=0.332851 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_H': 'same', 'alpha_W': 0.0, 'beta_loss': 'frobenius', 'init': 'nndsvda', 'l1_ratio': 0.0, 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'shuffle': False, 'solver': 'mu', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| init | "nndsvda" | "nndsvda" | "nndsvda" |
| l1_ratio | 0.0 | 0.0 | 0.0 |
| max_iter | 200 | 200 | 200 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| shuffle | false | false | false |
| solver | "mu" | "mu" | "mu" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### nmf / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.nmf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 268.6 | 268.6..268.6 | 1 | - | - | - | 386.1 | - | relative_reconstruction_error=0.089097 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 260.3 | 260.3..260.3 | 1 | - | - | - | 383.6 | - | relative_reconstruction_error=0.089097 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 12.4 | 12.4..12.4 | 1 | 21.592 | 20.925 | - | 142.6 | - | relative_reconstruction_error=0.089097 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha_H': 'same', 'alpha_W': 0.0, 'beta_loss': 'frobenius', 'init': 'nndsvda', 'l1_ratio': 0.0, 'max_iter': 200, 'n_components': 8, 'random_state': 7, 'shuffle': False, 'solver': 'mu', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| init | "nndsvda" | "nndsvda" | "nndsvda" |
| l1_ratio | 0.0 | 0.0 | 0.0 |
| max_iter | 200 | 200 | 200 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| shuffle | false | false | false |
| solver | "mu" | "mu" | "mu" |
| tol | 0.0001 | 0.0001 | 0.0001 |

### ocsvm / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.ocsvm.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'coef0': 0.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'nu': 0.1, 'shrinking': True, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### ocsvm / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.ocsvm.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'coef0': 0.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'nu': 0.1, 'shrinking': True, 'tol': 0.001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0.0 | 0.0 | 0.0 |
| degree | 3 | 3 | 3 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| kernel | "rbf" | "rbf" | "rbf" |
| max_iter | -1 | -1 | -1 |
| tol | 0.001 | 0.001 | 0.001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### optics / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.optics.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'algorithm': 'auto', 'cluster_method': 'xi', 'leaf_size': 30, 'max_eps': inf, 'metric': 'minkowski', 'min_samples': 10, 'p': 2, 'predecessor_correction': True, 'xi': 0.05}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

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

accepted difference: ours-fast eps: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast min_cluster_size: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu eps: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu min_cluster_size: None on both ours and sklearn-cpu: the same documented setting in both signatures

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### optics / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.optics.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'algorithm': 'auto', 'cluster_method': 'xi', 'leaf_size': 30, 'max_eps': inf, 'metric': 'minkowski', 'min_samples': 10, 'p': 2, 'predecessor_correction': True, 'xi': 0.05}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

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

accepted difference: ours-fast eps: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast min_cluster_size: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu eps: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu min_cluster_size: None on both ours and sklearn-cpu: the same documented setting in both signatures

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### pa-clf / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.pa-clf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 478.5 | 478.5..478.5 | 1 | - | - | - | 477.3 | - | accuracy=0.908500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 388.0 | 388.0..388.0 | 1 | - | - | - | 475.0 | - | accuracy=0.908500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 15.2 | 15.2..15.2 | 1 | 31.539 | 25.578 | - | 145.9 | - | accuracy=0.907500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'fit_intercept': True, 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### pa-clf / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.pa-clf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 49.5 | 49.5..49.5 | 1 | - | - | - | 469.8 | - | accuracy=0.324500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 80.1 | 80.1..80.1 | 1 | - | - | - | 467.0 | - | accuracy=0.324500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 22.316 | 36.133 | - | 141.4 | - | accuracy=0.680000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'fit_intercept': True, 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### pa-reg / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.pa-reg.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 489.9 | 489.9..489.9 | 1 | - | - | - | 478.2 | - | finite=True, r2=-0.783304, rmse=1.060094 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 391.9 | 391.9..391.9 | 1 | - | - | - | 473.5 | - | finite=True, r2=-0.783304, rmse=1.060094 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 19.9 | 19.9..19.9 | 1 | 24.630 | 19.706 | - | 148.4 | - | finite=True, r2=-0.524715, rmse=0.980225 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### pa-reg / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.pa-reg.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 50.7 | 50.7..50.7 | 1 | - | - | - | 470.9 | - | finite=True, r2=0.908105, rmse=4.428049 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 84.7 | 84.7..84.7 | 1 | - | - | - | 467.4 | - | finite=True, r2=0.908105, rmse=4.428049 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.8 | 1.8..1.8 | 1 | 27.857 | 46.517 | - | 143.4 | - | finite=True, r2=0.905999, rmse=4.478495 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'C': 1.0, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### pagerank / istella (rows 2000, shape X 2000x220; indices 30756; indices2 6214; indptr 2001; indptr2 2001; y 2000)

race: failed, driver rc 3, log `logs/algos.pagerank.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| networkx-cpu | networkx | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| alpha | 0.85 | 0.85 | 0.85 |
| max_iter | 100 | 100 | 100 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: networkx-cpu (networkx): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### pagerank / taxi (rows 2000, shape X 2000x11; indices 26186; indices2 5530; indptr 2001; indptr2 2001; y 2000)

race: failed, driver rc 3, log `logs/algos.pagerank.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| networkx-cpu | networkx | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.85, 'max_iter': 100, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: ours takes a dense adjacency matrix (its class's contract), built from the CSR graph before the clock; networkx and cuGraph take the graph itself

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | networkx-cpu | ours | ours-fast |
|---|---||---|---||---|---|
| library (source) | networkx (declared) | mojolearn (declared) | mojolearn (declared) |
| alpha | 0.85 | 0.85 | 0.85 |
| max_iter | 100 | 100 | 100 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: networkx-cpu (networkx): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### perceptron / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.perceptron.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 313.1 | 313.1..313.1 | 1 | - | - | - | 477.8 | - | accuracy=0.883500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 262.0 | 262.0..262.0 | 1 | - | - | - | 473.7 | - | accuracy=0.883500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7.5 | 7.5..7.5 | 1 | 41.892 | 35.049 | - | 146.0 | - | accuracy=0.914000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'early_stopping': False, 'eta0': 1.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
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

### perceptron / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.perceptron.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 47.6 | 47.6..47.6 | 1 | - | - | - | 468.9 | - | accuracy=0.404000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 82.0 | 82.0..82.0 | 1 | - | - | - | 468.4 | - | accuracy=0.404000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.9 | 1.9..1.9 | 1 | 24.639 | 42.405 | - | 143.7 | - | accuracy=0.590000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'early_stopping': False, 'eta0': 1.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'max_iter': 20, 'n_iter_no_change': 5, 'random_state': 7, 'shuffle': True, 'tol': None, 'validation_fraction': 0.1}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
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

### pls-canonical / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.pls-canonical.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'algorithm': 'nipals', 'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "nipals" | "nipals" | "nipals" |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### pls-canonical / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.pls-canonical.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'algorithm': 'nipals', 'max_iter': 500, 'n_components': 2, 'scale': True, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| algorithm | "nipals" | "nipals" | "nipals" |
| max_iter | 500 | 500 | 500 |
| n_components | 2 | 2 | 2 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### pls / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.pls.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'max_iter': 500, 'n_components': 4, 'scale': True, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 4 | 4 | 4 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### pls / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.pls.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'max_iter': 500, 'n_components': 4, 'scale': True, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| max_iter | 500 | 500 | 500 |
| n_components | 4 | 4 | 4 |
| tol | 1e-06 | 1e-06 | 1e-06 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### poisson / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.poisson.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### poisson / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.poisson.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 100, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### poly-count-sketch / istella (rows 2000, shape X 2000x220; Xq 1000x220; y 2000; yq 1000)

race: done, driver rc 0, log `logs/algos.poly-count-sketch.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 334.7 | - | kernel_rel_error=0.081609 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.3 | 0.3..0.3 | 1 | - | - | - | 340.8 | - | kernel_rel_error=0.081609 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.4 | 0.4..0.4 | 1 | 0.775 | 0.783 | - | 159.0 | - | kernel_rel_error=0.081609 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'coef0': 0, 'degree': 2, 'gamma': 1.0, 'n_components': 256, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0 | 0 | 0 |
| degree | 2 | 2 | 2 |
| gamma | 0.004545454545454545 | 0.004545454545454545 | 0.004545454545454545 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### poly-count-sketch / taxi (rows 2000, shape X 2000x11; Xq 1000x11; y 2000; yq 1000)

race: done, driver rc 0, log `logs/algos.poly-count-sketch.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 0.3 | 0.3..0.3 | 1 | - | - | - | 333.2 | - | kernel_rel_error=0.098761 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 0.3 | 0.3..0.3 | 1 | - | - | - | 331.7 | - | kernel_rel_error=0.098761 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.4 | 0.4..0.4 | 1 | 0.605 | 0.578 | - | 164.7 | - | kernel_rel_error=0.098761 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'coef0': 0, 'degree': 2, 'gamma': 1.0, 'n_components': 256, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| coef0 | 0 | 0 | 0 |
| degree | 2 | 2 | 2 |
| gamma | 0.09090909090909091 | 0.09090909090909091 | 0.09090909090909091 |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### quantile / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.quantile.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 5000 | 5000 | - |
| solver | "highs" | "highs" | "highs" |
| tol | 0.0001 | 0.0001 | - |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### quantile / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.quantile.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'quantile': 0.5, 'solver': 'highs'}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: solver: 'highs' passed to both; ours accepts the name and always runs ADMM (max_iter=5000, tol=1e-4, ours only), scikit-learn solves the HiGHS LP

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 5000 | 5000 | - |
| solver | "highs" | "highs" | "highs" |
| tol | 0.0001 | 0.0001 | - |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### randomized-svd / istella (rows 2000, shape X 1800x220; Xq 200x220)

race: done, driver rc 0, log `logs/algos.randomized-svd.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 33.2 | 33.2..33.2 | 1 | - | - | - | 364.8 | - | relative_reconstruction_error=0.0001122 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 32.1 | 32.1..32.1 | 1 | - | - | - | 362.2 | - | relative_reconstruction_error=0.0001122 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.9 | 1.9..1.9 | 1 | 17.784 | 17.237 | - | 127.9 | - | relative_reconstruction_error=0.0001122 | - | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 161.8 | 161.8..161.8 | 1 | 0.205 | 0.199 | - | 471.0 | 40.5 | relative_reconstruction_error=0.0001122 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) | torch (declared) |
| n_components | 8 | 8 | 8 | 8 |
| n_iter | 4 | 4 | 4 | 4 |
| seed | 7 | 7 | 7 | 7 |

### randomized-svd / taxi (rows 2000, shape X 1800x11; Xq 200x11)

race: done, driver rc 0, log `logs/algos.randomized-svd.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 23.5 | 23.5..23.5 | 1 | - | - | - | 353.2 | - | relative_reconstruction_error=0.027111 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 23.5 | 23.5..23.5 | 1 | - | - | - | 354.8 | - | relative_reconstruction_error=0.027111 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.2 | 1.2..1.2 | 1 | 19.941 | 19.932 | - | 125.9 | - | relative_reconstruction_error=0.027111 | - | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 98.0 | 98.0..98.0 | 1 | 0.240 | 0.240 | - | 1465.1 | 1032.5 | relative_reconstruction_error=0.027111 | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: {'n_components': 8, 'n_iter': 4, 'n_oversamples': 10, 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: torch-gpu is torch.svd_lowrank(q=18, niter=4), its randomized range finder

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu | torch-gpu |
|---|---||---|---||---|---||---|---|
| library (source) | mojolearn (declared) | mojolearn (declared) | sklearn (declared) | torch (declared) |
| n_components | 8 | 8 | 8 | 8 |
| n_iter | 4 | 4 | 4 | 4 |
| seed | 7 | 7 | 7 | 7 |

### ridge-clf / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.ridge-clf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 242.3 | 242.3..242.3 | 1 | - | - | - | 478.4 | - | accuracy=0.909000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 216.8 | 216.8..216.8 | 1 | - | - | - | 474.9 | - | accuracy=0.909000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.1 | 2.1..2.1 | 1 | 113.532 | 101.563 | - | 149.1 | - | accuracy=0.909000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_intercept': True, 'positive': False, 'random_state': 7, 'solver': 'auto', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### ridge-clf / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.ridge-clf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2.7 | 2.7..2.7 | 1 | - | - | - | 471.4 | - | accuracy=0.762500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2.9 | 2.9..2.9 | 1 | - | - | - | 469.8 | - | accuracy=0.762500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.2 | 1.2..1.2 | 1 | 2.305 | 2.429 | - | 142.9 | - | accuracy=0.762500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'fit_intercept': True, 'positive': False, 'random_state': 7, 'solver': 'auto', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

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

### ridge-cv / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.ridge-cv.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### ridge-cv / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.ridge-cv.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha_per_target': False, 'alphas': [0.001, 0.01, 0.1, 1.0, 10.0], 'cv': 5, 'fit_intercept': True}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### sgd-clf / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-clf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 324.1 | 324.1..324.1 | 1 | - | - | - | 475.9 | - | accuracy=0.916000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 291.4 | 291.4..291.4 | 1 | - | - | - | 475.1 | - | accuracy=0.916000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7.3 | 7.3..7.3 | 1 | 44.280 | 39.816 | - | 145.3 | - | accuracy=0.917500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs, not max_iter

mismatch: learning rate: scikit-learn and ours 'optimal' (1 / (alpha (t + t0))); cuML has no 'optimal' schedule and runs 'constant' eta0=0.001 (its default)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| epsilon | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "optimal" | "optimal" | "optimal" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 20 | 20 | 20 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-clf / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-clf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.7 | 54.7..54.7 | 1 | - | - | - | 473.3 | - | accuracy=0.637500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 80.5 | 80.5..80.5 | 1 | - | - | - | 469.6 | - | accuracy=0.637500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.8 | 1.8..1.8 | 1 | 29.622 | 43.572 | - | 144.4 | - | accuracy=0.731000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'hinge', 'max_iter': 20, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096); scikit-learn and ours are per-sample SGD (the reference)

mismatch: cuML reads epochs, not max_iter

mismatch: learning rate: scikit-learn and ours 'optimal' (1 / (alpha (t + t0))); cuML has no 'optimal' schedule and runs 'constant' eta0=0.001 (its default)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| class_weight | null | null | null |
| epsilon | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "optimal" | "optimal" | "optimal" |
| loss | "hinge" | "hinge" | "hinge" |
| max_iter | 20 | 20 | 20 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast class_weight: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu class_weight: None on both ours and sklearn-cpu: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-ocsvm / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 323.2 | 323.2..323.2 | 1 | - | - | - | 475.6 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 291.3 | 291.3..291.3 | 1 | - | - | - | 473.0 | - | fraction_flagged=0.000000, jaccard_vs_sklearn=0.000000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6.5 | 6.5..6.5 | 1 | 49.879 | 44.965 | - | 146.3 | - | fraction_flagged=0.013500, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'average': False, 'eta0': 0.0, 'fit_intercept': True, 'learning_rate': 'optimal', 'max_iter': 20, 'nu': 0.1, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| learning_rate | "optimal" | "optimal" | "optimal" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-ocsvm / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-ocsvm.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 48.0 | 48.0..48.0 | 1 | - | - | - | 470.7 | - | fraction_flagged=0.110000, jaccard_vs_sklearn=0.189427 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 80.3 | 80.3..80.3 | 1 | - | - | - | 466.8 | - | fraction_flagged=0.110000, jaccard_vs_sklearn=0.189427 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.3 | 1.3..1.3 | 1 | 37.383 | 62.523 | - | 142.9 | - | fraction_flagged=0.025000, jaccard_vs_sklearn=1.000000 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'average': False, 'eta0': 0.0, 'fit_intercept': True, 'learning_rate': 'optimal', 'max_iter': 20, 'nu': 0.1, 'power_t': 0.5, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| fit_intercept | true | true | true |
| learning_rate | "optimal" | "optimal" | "optimal" |
| max_iter | 20 | 20 | 20 |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-reg / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-reg.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 348.7 | 348.7..348.7 | 1 | - | - | - | 477.6 | - | finite=True, r2=-4.301e+07, rmse=5206.381796 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 260.2 | 260.2..260.2 | 1 | - | - | - | 472.5 | - | finite=True, r2=-4.301e+07, rmse=5206.367298 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 14.1 | 14.1..14.1 | 1 | 24.747 | 18.467 | - | 147.4 | - | finite=True, r2=-7.938e+07, rmse=7072.558134 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.01, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'invscaling', 'loss': 'squared_error', 'max_iter': 20, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "invscaling" | "invscaling" | "invscaling" |
| loss | "squared_error" | "squared_error" | "squared_error" |
| max_iter | 20 | 20 | 20 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### sgd-reg / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sgd-reg.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.7 | 54.7..54.7 | 1 | - | - | - | 467.0 | - | finite=True, r2=0.930328, rmse=3.855633 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 81.0 | 81.0..81.0 | 1 | - | - | - | 468.1 | - | finite=True, r2=0.930328, rmse=3.855633 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.9 | 1.9..1.9 | 1 | 28.931 | 42.820 | - | 144.5 | - | finite=True, r2=0.930762, rmse=3.843604 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 0.0001, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.01, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'invscaling', 'loss': 'squared_error', 'max_iter': 20, 'n_iter_no_change': 5, 'penalty': 'l2', 'power_t': 0.25, 'random_state': 7, 'shuffle': True, 'tol': None}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: cuML MBSGD is mini-batch SGD (batch_size 4096)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| epsilon | 0.1 | 0.1 | 0.1 |
| fit_intercept | true | true | true |
| l1_ratio | 0.15 | 0.15 | 0.15 |
| learning_rate | "invscaling" | "invscaling" | "invscaling" |
| loss | "squared_error" | "squared_error" | "squared_error" |
| max_iter | 20 | 20 | 20 |
| penalty | "l2" | "l2" | "l2" |
| seed | 7 | 7 | 7 |
| shuffle | true | true | true |
| tol | null | null | null |

accepted difference: ours-fast tol: None on both ours and ours-fast: the same documented setting in both signatures

accepted difference: sklearn-cpu tol: None on both ours and sklearn-cpu: the same documented setting in both signatures

### skewed-chi2 / istella (rows 2000, shape X 2000x220; Xq 1000x220; y 2000; yq 1000)

race: done, driver rc 0, log `logs/algos.skewed-chi2.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.1 | 7.1..7.1 | 1 | - | - | - | 344.7 | - | kernel_rel_error=0.740524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 7.4 | 7.4..7.4 | 1 | - | - | - | 342.1 | - | kernel_rel_error=0.740524 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.1 | 1.1..1.1 | 1 | 6.387 | 6.616 | - | 156.3 | - | kernel_rel_error=0.740524 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_components': 256, 'random_state': 7, 'skewedness': 1.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### skewed-chi2 / taxi (rows 2000, shape X 2000x11; Xq 1000x11; y 2000; yq 1000)

race: done, driver rc 0, log `logs/algos.skewed-chi2.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1.6 | 1.6..1.6 | 1 | - | - | - | 325.3 | - | kernel_rel_error=0.037983 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2.0 | 2.0..2.0 | 1 | - | - | - | 325.8 | - | kernel_rel_error=0.037983 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.4 | 0.4..0.4 | 1 | 4.094 | 4.987 | - | 147.3 | - | kernel_rel_error=0.037983 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'n_components': 256, 'random_state': 7, 'skewedness': 1.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| n_components | 256 | 256 | 256 |
| seed | 7 | 7 | 7 |

### sparse-pca / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sparse-pca.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4449.6 | 4449.6..4449.6 | 1 | - | - | - | 405.5 | - | component_sparsity=0.509659, relative_reconstruction_error=0.675875 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4058.8 | 4058.8..4058.8 | 1 | - | - | - | 391.1 | - | component_sparsity=0.509659, relative_reconstruction_error=0.675875 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 4766.1 | 4766.1..4766.1 | 1 | 0.934 | 0.852 | - | 173.8 | - | component_sparsity=0.509659, relative_reconstruction_error=0.675875 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'max_iter': 100, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 1e-06 | 1e-06 | 1e-06 |

### sparse-pca / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/algos.sparse-pca.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1800.5 | 1800.5..1800.5 | 1 | - | - | - | 361.3 | - | component_sparsity=0.693182, relative_reconstruction_error=0.282172 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1731.5 | 1731.5..1731.5 | 1 | - | - | - | 361.8 | - | component_sparsity=0.693182, relative_reconstruction_error=0.282172 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7445.2 | 7445.2..7445.2 | 1 | 0.242 | 0.233 | - | 164.1 | - | component_sparsity=0.693182, relative_reconstruction_error=0.282172 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'alpha': 1.0, 'max_iter': 100, 'method': 'lars', 'n_components': 8, 'random_state': 7, 'ridge_alpha': 0.01, 'tol': 1e-06}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 1.0 | 1.0 | 1.0 |
| max_iter | 100 | 100 | 100 |
| n_components | 8 | 8 | 8 |
| seed | 7 | 7 | 7 |
| tol | 1e-06 | 1e-06 | 1e-06 |

### sparse-rp / istella (rows 2000, shape X 1800x220; Xq 200x220)

race: done, driver rc 0, log `logs/algos.sparse-rp.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.1 | 3.1..3.1 | 1 | - | - | - | 332.2 | - | mean_abs_distortion=0.077581 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.6 | 4.6..4.6 | 1 | - | - | - | 330.6 | - | mean_abs_distortion=0.077581 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.0 | 1.0..1.0 | 1 | 3.229 | 4.806 | - | 147.3 | - | mean_abs_distortion=0.288047 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 'half', 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 110 | 110 | 110 |
| seed | 7 | 7 | 7 |

### sparse-rp / taxi (rows 2000, shape X 1800x11; Xq 200x11)

race: done, driver rc 0, log `logs/algos.sparse-rp.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3.7 | 3.7..3.7 | 1 | - | - | - | 321.5 | - | mean_abs_distortion=0.287559 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5.0 | 5.0..5.0 | 1 | - | - | - | 320.6 | - | mean_abs_distortion=0.287559 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 0.5 | 0.5..0.5 | 1 | 7.285 | 9.656 | - | 142.9 | - | mean_abs_distortion=0.460698 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: {'compute_inverse_components': False, 'dense_output': False, 'density': 'auto', 'eps': 0.1, 'n_components': 'half', 'random_state': 7}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| eps | 0.1 | 0.1 | 0.1 |
| n_components | 5 | 5 | 5 |
| seed | 7 | 7 | 7 |

### svgp / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.svgp.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| gpytorch-gpu | gpytorch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| gpytorch-cpu | gpytorch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | gpytorch-cpu | gpytorch-gpu | ours | ours-fast |
|---|---||---|---||---|---||---|---|
| library (source) | gpytorch (declared) | gpytorch (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | - | - |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### svgp / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.svgp.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| gpytorch-gpu | gpytorch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| gpytorch-cpu | gpytorch | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (2): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'jitter': 1e-06, 'kernel_variance': 1.0, 'lengthscale': 1.0, 'n_inducing': 512, 'noise_variance': 1.0}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

mismatch: no seed on any arm: nothing is drawn (fixed inducing points, closed form)

mismatch: jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) only when a factorization fails

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | gpytorch-cpu | gpytorch-gpu | ours | ours-fast |
|---|---||---|---||---|---||---|---|
| library (source) | gpytorch (declared) | gpytorch (declared) | mojolearn (declared) | mojolearn (declared) |
| seed | 7 | 7 | - | - |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### tweedie / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.tweedie.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'link': 'log', 'max_iter': 100, 'power': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

### tweedie / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: failed, driver rc 3, log `logs/algos.tweedie.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(params_refused: "parameters do not match (3): ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none; ours-fast (mojolearn): no seed parameter read back; list it ) |

settings: {'alpha': 0.0001, 'fit_intercept': True, 'link': 'log', 'max_iter': 100, 'power': 1.5, 'solver': 'lbfgs', 'tol': 0.0001}. Rows (the full board; this run caps them at --rows 2000): None. Timed: None.

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): REFUSED

| parameter | ours | ours-fast | sklearn-cpu |
|---|---||---|---||---|---|
| library (source) | mojolearn (get_params) | mojolearn (get_params) | sklearn (get_params) |
| alpha | 0.0001 | 0.0001 | 0.0001 |
| fit_intercept | true | true | true |
| max_iter | 100 | 100 | 100 |
| solver | "lbfgs" | "lbfgs" | "lbfgs" |
| tol | 0.0001 | 0.0001 | 0.0001 |

REFUSED: ours (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: ours-fast (mojolearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

REFUSED: sklearn-cpu (sklearn): no seed parameter read back; list it in EXCEPTIONS with the reason if the library has none

## Not covered by this board

- Classical, wave 2: RadiusNeighbors, the preprocessing scalers, HDBSCAN's prediction data, Cholesky and the parallel_* and Distributed* wrappers are public and not raced here; taxi-derived time series are not used (the ARIMA and ExponentialSmoothing lanes fit seeded synthetic series, as the repo's own ARIMA quality work does).
- Classical, wave 2, not planned on this vendor: cuML and cuVS: CUDA only; no Apple build exists.
- Classical, wave 2, not planned on this vendor: faiss-gpu: CUDA only; faiss-cpu is the arm on this box.
- Inference, trees: a single-row latency batch is not timed (the batches are the held-out split and 1,000,000 training rows); ONNX, Treelite and other export paths are not raced.
- Inference, classical: the classical2 family's predict calls (the linear models, GaussianMixture, SVR, KernelRidge and others in tools/bench_board_more.py) are timed as those lanes define their clocks, not as a separate inference cell; svc times predict, not decision_function.
- Our CPU tier: `--no-cpu-arm` was passed, so no `ours-cpu` arm ran.
- Our CPU tier, no ours-cpu arm: neural mamba1-infer, mamba2-infer, mamba3-infer, mlp-infer, samba-infer, transformer-infer: its `ours` arm already IS the CPU path (the public *Inference class runs on the host binding).
- Our CPU tier: a GBDT configuration the host side does not restate refuses by name in its ours-cpu cell (python/mojolearn/host_surface.py NO_CPU_PATH lists them), and a FAST-only run (`--modes fast`) has no ours-cpu arm: the host bindings build IDENTICAL only.
- Memory: GPU memory on Apple has no per-process counter (Metal buffers are inside the host footprint); the trees driver runs every arm in one process, so its GPU figure is the process total; a figure taken at the round's end misses a buffer freed inside the round; inference cells carry memory only on the classical lanes.
- Neural: The Mamba opponents are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton kernels, which the board does not install; a Mamba ratio here is against a reference implementation, not a deployment kernel.
- Neural: The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged `lengths` and the carried-state forward are public and not raced; only a zero-state forward is.
- Neural: SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; MLPInference (its CPU forward) and the training step are.
- Neural: The GPT-3-small target shape is not on the board (the LM lanes use the smaller control shape so one shape runs on every box, a 16 GB Mac included).
- Neural, not planned on this vendor: torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul mode; torch on MPS accepts the flag and changes nothing
- Neural, not planned on this vendor: torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is the pure-PyTorch reference scan, a per-token Python loop that torch.compile would unroll L times; mamba1 races the eager arms only

