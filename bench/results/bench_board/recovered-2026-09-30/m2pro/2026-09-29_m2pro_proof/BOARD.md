# mojolearn benchmark board

Generated 2026-09-29T14:27:35Z from `board.json` (schema `mojolearn-bench-board/1`).

> SMOKE RUN: `--rows 2000` is below the 1,000,000-row tree floor or the classical lane shapes; `--neural-shape small` is a plumbing shape. These numbers are plumbing checks, not results.

## Box

| field | value |
|---|---|
| vendor / API | apple / metal |
| GPU | Apple M2 Pro |
| GPU driver | macOS 26.7 |
| CPU | Apple M2 Pro (12 logical cores) |
| memory bytes | 34359738368 |
| OS | macOS 26.7 |
| Python | 3.13.15 CPython |
| mojolearn | 0.8.25 (wheel mojolearn-0.8.25-py3-none-macosx_11_0_arm64.whl, sha256 ee1187c950c29e791c906cb3beb1c45206e4d8534fea897bc839db4505b38548) |
| script commit | c64a3ce629b0a9cc5c74861a5a6242273142c702 |
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

Races: 464 planned, 46 done, 1 failed, 417 pending. Cells: 170 (REFUSED 7, ok 163).

## Quality at a glance

Per lane and dataset: our FAST value, our IDENTICAL value, our CPU IDENTICAL value, and each opponent's.

| family | lane | dataset | metric | ours FAST | ours IDENTICAL | ours CPU | opponents |
|---|---|---|---|---|---|---|---|
| classical | dbscan | istella | n_clusters | 179 | 179 | - | sklearn-cpu 179 |
| classical | dbscan | istella | noise_fraction | 0.564500 | 0.564500 | - | sklearn-cpu 0.564500 |
| classical | dbscan | istella | rows | 2000 | 2000 | - | sklearn-cpu 2000 |
| classical | dbscan | istella | ari_vs_ours (1 is our partition exactly) | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | istella | noise_agreement_vs_ours | 1.000000 | - | - | sklearn-cpu 1.000000 |
| classical | dbscan | taxi | n_clusters | 10 | 10 | - | sklearn-cpu 10 |
| classical | dbscan | taxi | noise_fraction | 0.005500 | 0.005500 | - | sklearn-cpu 0.005500 |
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
| classical | kde | istella | mean_log_likelihood (higher is better) | -398.822031 | -398.822047 | - | sklearn-cpu -397.566925 |
| classical | kde | istella | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kde | taxi | mean_log_likelihood (higher is better) | -14.818538 | -14.818538 | - | sklearn-cpu -14.818536 |
| classical | kde | taxi | rows_without_density | 0 | 0 | - | sklearn-cpu 0 |
| classical | kmeans | istella | inertia (lower is better) | 4.136e+14 | 4.136e+14 | - | sklearn-cpu 4.115e+14; torch-gpu 4.117e+14 |
| classical | kmeans | istella | inertia_over_ours | 1.000000 | 1.000000 | - | sklearn-cpu 0.994966; torch-gpu 0.995480 |
| classical | kmeans | istella | n_iter | 8 | 8 | - | sklearn-cpu 5; torch-gpu 6 |
| classical | kmeans | taxi | inertia (lower is better) | 25888.434366 | 25888.434366 | - | sklearn-cpu 27617.726175; torch-gpu 27982.170005 |
| classical | kmeans | taxi | inertia_over_ours | 1.000000 | 1.000000 | - | sklearn-cpu 1.066798; torch-gpu 1.080875 |
| classical | kmeans | taxi | n_iter | 14 | 14 | - | sklearn-cpu 67; torch-gpu 26 |
| classical | knn | istella | recall_at_k (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu 1.000000 |
| classical | knn | istella | rows_with_repeated_ids | 0 | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | knn | taxi | recall_at_k (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu 1.000000 |
| classical | knn | taxi | rows_with_repeated_ids | 0 | 0 | - | sklearn-cpu 0; torch-gpu 0 |
| classical | ols | istella | r2 (higher is better) | -24.390212 | -24.129541 | - | sklearn-cpu -0.027117; torch-gpu - |
| classical | ols | istella | rmse (lower is better) | 4.771147 | 4.746592 | - | sklearn-cpu 0.959621; torch-gpu - |
| classical | ols | taxi | r2 (higher is better) | 0.889542 | 0.889788 | - | sklearn-cpu 0.907724; torch-gpu - |
| classical | ols | taxi | rmse (lower is better) | 6.587113 | 6.579774 | - | sklearn-cpu 6.020627; torch-gpu - |
| classical | pca | istella | explained_variance_ratio_sum (higher is better) | 1.000000 | 1.000000 | - | sklearn-cpu 1.000000; torch-gpu - |
| classical | pca | taxi | explained_variance_ratio_sum (higher is better) | 1.000001 | 1.000001 | - | sklearn-cpu 1.000000; torch-gpu - |
| classical | svc | istella | accuracy (higher is better) | 0.855469 | 0.855469 | - | sklearn-cpu 0.855469 |
| classical | svc | istella | n_support | 94 | 94 | - | sklearn-cpu 94 |
| classical | svc | taxi | accuracy (higher is better) | 0.765625 | 0.765625 | - | sklearn-cpu 0.765625 |
| classical | svc | taxi | n_support | 151 | 151 | - | sklearn-cpu 151 |
| classical2 | gmm | taxi | bic (lower is better) | -65151.291133 | -68064.910372 | - | sklearn-cpu -69061.429868 |
| classical2 | gmm | taxi | mean_log_likelihood (higher is better) | 12.570116 | 12.918593 | - | sklearn-cpu 13.245844 |
| classical2 | gmm | taxi | n_iter | 11 | 25 | - | sklearn-cpu 100 |
| classical2 | linearsvc | istella | accuracy (higher is better) | 0.912500 | 0.913500 | - | sklearn-cpu 0.913500 |
| classical2 | linearsvc | taxi | accuracy (higher is better) | 0.763000 | 0.763000 | - | sklearn-cpu 0.762500 |
| classical2 | logreg | istella | accuracy (higher is better) | 0.923000 | 0.923500 | - | sklearn-cpu 0.923500 |
| classical2 | logreg | istella | logloss (lower is better) | 0.204828 | 0.204798 | - | sklearn-cpu 0.204942 |
| classical2 | logreg | istella | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | logreg | taxi | accuracy (higher is better) | 0.763500 | 0.763500 | - | sklearn-cpu 0.763500 |
| classical2 | logreg | taxi | logloss (lower is better) | 0.547874 | 0.547874 | - | sklearn-cpu 0.547867 |
| classical2 | logreg | taxi | nonfinite_proba_rows | 0 | 0 | - | sklearn-cpu 0 |
| classical2 | spectral-embedding | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.858681 | 0.858681 | - | sklearn-cpu 0.858691 |
| classical2 | spectral-embedding | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.708797 | 0.690399 | - | sklearn-cpu 0.687460 |
| classical2 | umap | istella | trustworthiness_k15 (higher is better, 1 at most) | 0.942379 | 0.943567 | - | umap-learn-cpu 0.945192; umap-learn-cpu-unseeded 0.942601 |
| classical2 | umap | taxi | trustworthiness_k15 (higher is better, 1 at most) | 0.973370 | 0.973542 | - | umap-learn-cpu 0.973861; umap-learn-cpu-unseeded 0.975341 |
| trees | et | istella | logloss (lower is better) | 0.532243 | 0.532243 | - | sklearn-et-cpu 0.542278; lightgbm-cpu 0.273911 |
| trees | et | istella | auc (higher is better) | 0.857018 | 0.857018 | - | sklearn-et-cpu 0.858858; lightgbm-cpu 0.877419 |
| trees | et | taxi | logloss (lower is better) | 0.644383 | 0.644383 | - | sklearn-et-cpu 0.652006; lightgbm-cpu 0.864346 |
| trees | et | taxi | auc (higher is better) | 0.573465 | 0.573465 | - | sklearn-et-cpu 0.566032; lightgbm-cpu 0.565876 |
| trees | gbdt-categorical | taxi | logloss (lower is better) | 0.999877 | 1.065861 | - | catboost-cpu 0.724699; xgboost-cpu 1.521580; lightgbm-cpu 0.921276 |
| trees | gbdt-categorical | taxi | auc (higher is better) | 0.502514 | 0.494784 | - | catboost-cpu 0.528209; xgboost-cpu 0.507573; lightgbm-cpu 0.556942 |
| trees | gbdt-depthwise | istella | logloss (lower is better) | 0.441845 | 0.434352 | - | catboost-cpu 0.355696; xgboost-cpu 0.614195 |
| trees | gbdt-depthwise | istella | auc (higher is better) | 0.833245 | 0.834435 | - | catboost-cpu 0.900776; xgboost-cpu 0.879129 |
| trees | gbdt-depthwise | taxi | logloss (lower is better) | 0.857163 | 0.888202 | - | catboost-cpu 1.036669; xgboost-cpu 0.872790 |
| trees | gbdt-depthwise | taxi | auc (higher is better) | 0.533606 | 0.533726 | - | catboost-cpu 0.552206; xgboost-cpu 0.551051 |
| trees | gbdt-lossguide | istella | logloss (lower is better) | 0.606902 | 0.615650 | - | catboost-cpu 0.355696; xgboost-cpu 0.614195; lightgbm-cpu 0.580275 |
| trees | gbdt-lossguide | istella | auc (higher is better) | 0.840519 | 0.839015 | - | catboost-cpu 0.900776; xgboost-cpu 0.879129; lightgbm-cpu 0.879389 |
| trees | gbdt-lossguide | taxi | logloss (lower is better) | 0.828055 | 0.838647 | - | catboost-cpu 1.036669; xgboost-cpu 0.872790; lightgbm-cpu 0.794800 |
| trees | gbdt-lossguide | taxi | auc (higher is better) | 0.533783 | 0.533833 | - | catboost-cpu 0.552206; xgboost-cpu 0.551051; lightgbm-cpu 0.551531 |
| trees | gbdt-multiclass | istella | mlogloss (lower is better) | 0.616408 | 0.615914 | - | catboost-cpu 0.597921; xgboost-cpu 0.900271; lightgbm-cpu 0.902228 |
| trees | gbdt-multiclass | istella | accuracy (higher is better) | 0.887012 | 0.887036 | - | catboost-cpu 0.886878; xgboost-cpu 0.888354; lightgbm-cpu 0.889184 |
| trees | gbdt-multiclass | taxi | mlogloss (lower is better) | 1.527569 | 1.527569 | - | catboost-cpu 1.461085; xgboost-cpu 2.296402; lightgbm-cpu 2.202995 |
| trees | gbdt-multiclass | taxi | accuracy (higher is better) | 0.431384 | 0.431384 | - | catboost-cpu 0.427406; xgboost-cpu 0.422622; lightgbm-cpu 0.417378 |
| trees | gbdt-ordered | istella | logloss (lower is better) | 0.438835 | 0.438932 | - | catboost-cpu 0.382400 |
| trees | gbdt-ordered | istella | auc (higher is better) | 0.891611 | 0.891597 | - | catboost-cpu 0.903468 |
| trees | gbdt-ordered | taxi | logloss (lower is better) | 0.788715 | 0.785560 | - | catboost-cpu 0.642938 |
| trees | gbdt-ordered | taxi | auc (higher is better) | 0.564937 | 0.565213 | - | catboost-cpu 0.563277 |
| trees | gbdt-rank-pairlogit | istella | ndcg10 (higher is better) | 0.513409 | 0.554901 | - | catboost-cpu 0.624304; xgboost-cpu 0.586953 |
| trees | gbdt-rank-pairlogit | istella | ndcg5 (higher is better) | 0.467814 | 0.499610 | - | catboost-cpu 0.561860; xgboost-cpu 0.530981 |
| trees | gbdt-rank-pairlogit | istella | map (higher is better) | 0.581099 | 0.636433 | - | catboost-cpu 0.730439; xgboost-cpu 0.698987 |
| trees | gbdt-rank-yetirank | istella | ndcg10 (higher is better) | 0.526958 | 0.556211 | - | catboost-cpu 0.601311; xgboost-cpu 0.579654; lightgbm-cpu 0.601041 |
| trees | gbdt-rank-yetirank | istella | ndcg5 (higher is better) | 0.477677 | 0.498282 | - | catboost-cpu 0.539947; xgboost-cpu 0.519897; lightgbm-cpu 0.540794 |
| trees | gbdt-rank-yetirank | istella | map (higher is better) | 0.603122 | 0.645419 | - | catboost-cpu 0.713065; xgboost-cpu 0.690516; lightgbm-cpu 0.715387 |
| trees | gbdt-symmetric-1000 | istella | logloss (lower is better) | 0.470812 | 0.436064 | - | catboost-cpu 0.352643 |
| trees | gbdt-symmetric-1000 | istella | auc (higher is better) | 0.834646 | 0.858086 | - | catboost-cpu 0.902796 |
| trees | gbdt-symmetric-1000 | taxi | logloss (lower is better) | 0.815915 | 0.814978 | - | catboost-cpu 0.803834 |
| trees | gbdt-symmetric-1000 | taxi | auc (higher is better) | 0.530109 | 0.529907 | - | catboost-cpu 0.547713 |
| trees | gbdt-symmetric | istella | logloss (lower is better) | 0.410556 | 0.379810 | - | catboost-cpu 0.305763 |
| trees | gbdt-symmetric | istella | auc (higher is better) | 0.834553 | 0.859575 | - | catboost-cpu 0.904749 |
| trees | gbdt-symmetric | taxi | logloss (lower is better) | 0.751517 | 0.752401 | - | catboost-cpu 0.703954 |
| trees | gbdt-symmetric | taxi | auc (higher is better) | 0.532717 | 0.532146 | - | catboost-cpu 0.548727 |
| trees | iforest | istella | auc (higher is better) | 0.763032 | 0.763032 | - | sklearn-iforest-cpu 0.792792 |
| trees | iforest | taxi | auc (higher is better) | 0.542833 | 0.542833 | - | sklearn-iforest-cpu 0.538780 |
| trees | rf | istella | logloss (lower is better) | 0.273691 | 0.273691 | - | sklearn-rf-cpu 0.273000; lightgbm-cpu 0.259049 |
| trees | rf | istella | auc (higher is better) | 0.885670 | 0.885670 | - | sklearn-rf-cpu 0.886759; lightgbm-cpu 0.884948 |
| trees | rf | taxi | logloss (lower is better) | 0.621278 | 0.621278 | - | sklearn-rf-cpu 0.593391; lightgbm-cpu 0.721738 |
| trees | rf | taxi | auc (higher is better) | 0.586569 | 0.586569 | - | sklearn-rf-cpu 0.587152; lightgbm-cpu 0.579249 |

## Trees

### et / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/et.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 120.6 | 120.6..120.6 | 1 | - | - | - | 3499.8 | - | auc=0.857018, logloss=0.532243 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 105.1 | 105.1..105.1 | 1 | - | - | - | 3913.5 | - | auc=0.857018, logloss=0.532243 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 91.1 | 91.1..91.1 | 1 | 1.323 | 1.153 | - | 3397.6 | - | auc=0.858858, logloss=0.542278 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 3376.3 | 3376.3..3376.3 | 1 | 0.036 | 0.031 | - | 14774.1 | - | auc=0.877419, logloss=0.273911 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:9042,ours:15839,ours-ab:15839,sklearn-et-cpu:15865 spread=0.4301 verdict=NOT-COMPARABLE`

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

### et / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/et.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 80.4 | 80.4..80.4 | 1 | - | - | - | 665.9 | - | auc=0.573465, logloss=0.644383 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 79.4 | 79.4..79.4 | 1 | - | - | - | 703.3 | - | auc=0.573465, logloss=0.644383 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-et-cpu | scikit-learn | cpu | opponent | 116.9 | 116.9..116.9 | 1 | 0.688 | 0.679 | - | 640.7 | - | auc=0.566032, logloss=0.652006 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 752.8 | 752.8..752.8 | 1 | 0.107 | 0.106 | - | 1698.5 | - | auc=0.565876, logloss=0.864346 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-et-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=et arms=lightgbm-cpu,ours,ours-ab,sklearn-et-cpu leaves=lightgbm-cpu:2045,ours:41616,ours-ab:41616,sklearn-et-cpu:40790 spread=0.9509 verdict=NOT-COMPARABLE`

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

### gbdt-categorical / taxi (rows 2000, shape taxicat-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-categorical.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 31216.7 | 31216.7..31216.7 | 1 | - | - | - | 671.4 | - | auc=0.494784, logloss=1.065861 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46258.3 | 46258.3..46258.3 | 1 | - | - | - | 916.3 | - | auc=0.502514, logloss=0.999877 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1935.8 | 1935.8..1935.8 | 1 | 16.126 | 23.896 | - | 1086.4 | - | auc=0.528209, logloss=0.724699 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 12834.7 | 12834.7..12834.7 | 1 | 2.432 | 3.604 | - | 1149.9 | - | auc=0.507573, logloss=1.521580 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 3785.5 | 3785.5..3785.5 | 1 | 8.246 | 12.220 | - | 1148.1 | - | auc=0.556942, logloss=0.921276 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-categorical arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:35931,lightgbm-cpu:21217,ours:48539,ours-ab:48723,xgboost-cpu:60035 spread=0.6466 verdict=NOT-COMPARABLE`

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
| scale_pos_weight | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 |
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
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6656.4 | 6656.4..6656.4 | 1 | - | - | - | 2517.0 | - | auc=0.834435, logloss=0.434352 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 8433.5 | 8433.5..8433.5 | 1 | - | - | - | 3435.0 | - | auc=0.833245, logloss=0.441845 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 2800.0 | 2800.0..2800.0 | 1 | 2.377 | 3.012 | - | 4288.3 | - | auc=0.900776, logloss=0.355696 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 1854.4 | 1854.4..1854.4 | 1 | 3.590 | 4.548 | - | 4757.7 | - | auc=0.879129, logloss=0.614195 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:20090,ours:18074,ours-ab:18561,xgboost-cpu:25485 spread=0.2908 verdict=NOT-COMPARABLE`

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
| scale_pos_weight | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 |
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
| mojolearn IDENTICAL | mojolearn | gpu | identical | 4172.0 | 4172.0..4172.0 | 1 | - | - | - | 577.8 | - | auc=0.533726, logloss=0.888202 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5033.5 | 5033.5..5033.5 | 1 | - | - | - | 658.7 | - | auc=0.533606, logloss=0.857163 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1158.7 | 1158.7..1158.7 | 1 | 3.601 | 4.344 | - | 727.7 | - | auc=0.552206, logloss=1.036669 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 1253.2 | 1253.2..1253.2 | 1 | 3.329 | 4.016 | - | 769.8 | - | auc=0.551051, logloss=0.872790 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-depthwise arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:34397,ours:15204,ours-ab:17297,xgboost-cpu:39528 spread=0.6154 verdict=NOT-COMPARABLE`

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
| scale_pos_weight | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 |
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
| mojolearn IDENTICAL | mojolearn | gpu | identical | 32863.3 | 32863.3..32863.3 | 1 | - | - | - | 2536.5 | - | auc=0.839015, logloss=0.615650 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 37295.5 | 37295.5..37295.5 | 1 | - | - | - | 3443.4 | - | auc=0.840519, logloss=0.606902 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 4433.4 | 4433.4..4433.4 | 1 | 7.413 | 8.412 | - | 4312.4 | - | auc=0.900776, logloss=0.355696 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 6601.5 | 6601.5..6601.5 | 1 | 4.978 | 5.650 | - | 4776.1 | - | auc=0.879129, logloss=0.614195 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 2903.9 | 2903.9..2903.9 | 1 | 11.317 | 12.843 | - | 4773.0 | - | auc=0.879389, logloss=0.580275 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:20090,lightgbm-cpu:12225,ours:46206,ours-ab:38939,xgboost-cpu:25485 spread=0.7354 verdict=NOT-COMPARABLE`

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
| scale_pos_weight | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 |
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
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12407.2 | 12407.2..12407.2 | 1 | - | - | - | 577.6 | - | auc=0.533833, logloss=0.838647 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 13615.1 | 13615.1..13615.1 | 1 | - | - | - | 657.2 | - | auc=0.533783, logloss=0.828055 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1476.2 | 1476.2..1476.2 | 1 | 8.405 | 9.223 | - | 718.7 | - | auc=0.552206, logloss=1.036669 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 8430.3 | 8430.3..8430.3 | 1 | 1.472 | 1.615 | - | 767.7 | - | auc=0.551051, logloss=0.872790 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 2350.4 | 2350.4..2350.4 | 1 | 5.279 | 5.793 | - | 773.7 | - | auc=0.551531, logloss=0.794800 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-lossguide arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:34397,lightgbm-cpu:13110,ours:18557,ours-ab:17210,xgboost-cpu:39528 spread=0.6683 verdict=NOT-COMPARABLE`

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
| scale_pos_weight | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 |
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
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12598.5 | 12598.5..12598.5 | 1 | - | - | - | 2942.5 | - | accuracy=0.887036, mlogloss=0.615914 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 12195.1 | 12195.1..12195.1 | 1 | - | - | - | 3911.1 | - | accuracy=0.887012, mlogloss=0.616408 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 10724.1 | 10724.1..10724.1 | 1 | 1.175 | 1.137 | - | 4701.1 | - | accuracy=0.886878, mlogloss=0.597921 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 8578.2 | 8578.2..8578.2 | 1 | 1.469 | 1.422 | - | 5247.9 | - | accuracy=0.888354, mlogloss=0.900271 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 11194.6 | 11194.6..11194.6 | 1 | 1.125 | 1.089 | - | 5220.6 | - | accuracy=0.889184, mlogloss=0.902228 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:125968,lightgbm-cpu:10339,ours:128000,ours-ab:128000,xgboost-cpu:19073 spread=0.9192 verdict=NOT-COMPARABLE`

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

### gbdt-multiclass / taxi (rows 2000, shape taximc-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-multiclass.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6103.0 | 6103.0..6103.0 | 1 | - | - | - | 788.2 | - | accuracy=0.431384, mlogloss=1.527569 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5114.1 | 5114.1..5114.1 | 1 | - | - | - | 882.4 | - | accuracy=0.431384, mlogloss=1.527569 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1271.1 | 1271.1..1271.1 | 1 | 4.802 | 4.024 | - | 968.0 | - | accuracy=0.427406, mlogloss=1.461085 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 5213.3 | 5213.3..5213.3 | 1 | 1.171 | 0.981 | - | 1048.2 | - | accuracy=0.422622, mlogloss=2.296402 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 10032.8 | 10032.8..10032.8 | 1 | 0.608 | 0.510 | - | 1058.9 | - | accuracy=0.417378, mlogloss=2.202995 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-multiclass arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:126260,lightgbm-cpu:13827,ours:128000,ours-ab:128000,xgboost-cpu:41770 spread=0.8920 verdict=NOT-COMPARABLE`

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

### gbdt-ordered / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/gbdt-ordered.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 21038.8 | 21038.8..21038.8 | 1 | - | - | - | 2878.0 | - | auc=0.891597, logloss=0.438932 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 20919.6 | 20919.6..20919.6 | 1 | - | - | - | 3786.6 | - | auc=0.891611, logloss=0.438835 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 11419.0 | 11419.0..11419.0 | 1 | 1.842 | 1.832 | - | 4444.4 | - | auc=0.903468, logloss=0.382400 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-ordered arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:126976,ours:127400,ours-ab:127400 spread=0.0033 verdict=COMPARABLE`

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) |
| boosting_type | "Ordered" | "Ordered" | "Ordered" |
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
| scale_pos_weight | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### gbdt-ordered / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-ordered.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 10197.4 | 10197.4..10197.4 | 1 | - | - | - | 707.4 | - | auc=0.565213, logloss=0.785560 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10328.6 | 10328.6..10328.6 | 1 | - | - | - | 768.5 | - | auc=0.564937, logloss=0.788715 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 1202.9 | 1202.9..1202.9 | 1 | 8.478 | 8.587 | - | 820.5 | - | auc=0.563277, logloss=0.642938 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-ordered arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:71722,ours:78814,ours-ab:79070 spread=0.0929 verdict=COMPARABLE`

config: NVIDIA gbm-bench, cat shared_params, ntrees 500, boosting_type 'Ordered' (https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py)

parameters (tools/bench_board_params.py, read back from each constructed arm; reference `ours`, seed 7): MATCHED

| parameter | catboost-cpu | ours | ours-ab |
|---|---||---|---||---|---|
| library (source) | catboost (get_params) | mojolearn (get_params) | mojolearn (get_params) |
| boosting_type | "Ordered" | "Ordered" | "Ordered" |
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
| scale_pos_weight | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### gbdt-rank-pairlogit / istella (rows 2000, shape istellarank-1928x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-pairlogit.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1287.6 | 1287.6..1287.6 | 1 | - | - | - | 2601.6 | - | map=0.636433, ndcg10=0.554901, ndcg5=0.499610 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 1068.6 | 1068.6..1068.6 | 1 | - | - | - | 3843.7 | - | map=0.581099, ndcg10=0.513409, ndcg5=0.467814 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 320.0 | 320.0..320.0 | 1 | 4.023 | 3.339 | - | 5069.7 | - | map=0.730439, ndcg10=0.624304, ndcg5=0.561860 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 263.2 | 263.2..263.2 | 1 | 4.892 | 4.060 | - | 5728.7 | - | map=0.698987, ndcg10=0.586953, ndcg5=0.530981 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-pairlogit arms=catboost-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6368,ours:6400,ours-ab:6400,xgboost-cpu:2868 spread=0.5519 verdict=NOT-COMPARABLE`

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

### gbdt-rank-yetirank / istella (rows 2000, shape istellarank-1928x220)

race: done, driver rc 0, log `raw/trees/gbdt-rank-yetirank.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9462.4 | 9462.4..9462.4 | 1 | - | - | - | 2601.6 | - | map=0.645419, ndcg10=0.556211, ndcg5=0.498282 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10723.8 | 10723.8..10723.8 | 1 | - | - | - | 3837.9 | - | map=0.603122, ndcg10=0.526958, ndcg5=0.477677 | yes | NOT-COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 420.6 | 420.6..420.6 | 1 | 22.495 | 25.494 | - | 5070.7 | - | map=0.713065, ndcg10=0.601311, ndcg5=0.539947 | yes | NOT-COMPARABLE | - | ok |
| xgboost-cpu | xgboost | cpu | opponent | 307.4 | 307.4..307.4 | 1 | 30.785 | 34.889 | - | 5707.7 | - | map=0.690516, ndcg10=0.579654, ndcg5=0.519897 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 494.5 | 494.5..494.5 | 1 | 19.134 | 21.685 | - | 5690.1 | - | map=0.715387, ndcg10=0.601041, ndcg5=0.540794 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu, xgboost-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-rank-yetirank arms=catboost-cpu,lightgbm-cpu,ours,ours-ab,xgboost-cpu leaves=catboost-cpu:6400,lightgbm-cpu:1890,ours:6400,ours-ab:6400,xgboost-cpu:3348 spread=0.7047 verdict=NOT-COMPARABLE`

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

### gbdt-symmetric-1000 / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 11173.7 | 11173.7..11173.7 | 1 | - | - | - | 2479.2 | - | auc=0.858086, logloss=0.436064 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10255.6 | 10255.6..10255.6 | 1 | - | - | - | 3377.1 | - | auc=0.834646, logloss=0.470812 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 5359.0 | 5359.0..5359.0 | 1 | 2.085 | 1.914 | - | 4277.0 | - | auc=0.902796, logloss=0.352643 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:255376,ours:256000,ours-ab:256000 spread=0.0024 verdict=COMPARABLE`

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
| scale_pos_weight | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### gbdt-symmetric-1000 / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric-1000.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 6173.1 | 6173.1..6173.1 | 1 | - | - | - | 535.3 | - | auc=0.529907, logloss=0.814978 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6247.8 | 6247.8..6247.8 | 1 | - | - | - | 622.3 | - | auc=0.530109, logloss=0.815915 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 803.0 | 803.0..803.0 | 1 | 7.688 | 7.781 | - | 683.1 | - | auc=0.547713, logloss=0.803834 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric-1000 arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:251672,ours:256000,ours-ab:256000 spread=0.0169 verdict=COMPARABLE`

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
| scale_pos_weight | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 |
| score_function | "Cosine" | "Cosine" | "Cosine" |
| seed | 7 | 7 | 7 |
| subsample | null | null | null |

accepted difference: ours-ab min_child_weight: ours takes min_child_hessian on Depthwise and Lossguide only; on the symmetric grower it is unset (no hessian floor, as CatBoost; XGBoost 0)

accepted difference: ours-ab min_split_gain: ours takes min_split_gain on Depthwise and Lossguide only; on the symmetric grower it is unset (CatBoost has none; XGBoost gamma 0, LightGBM 0)

accepted difference: ours-ab subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

accepted difference: catboost-cpu subsample: no row sampling on any arm: ours and CatBoost bootstrap_type 'No' (neither accepts subsample beside it, so it stays unset), XGBoost and LightGBM subsample 1.0

### gbdt-symmetric / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/gbdt-symmetric.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 5859.2 | 5859.2..5859.2 | 1 | - | - | - | 2445.0 | - | auc=0.859575, logloss=0.379810 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 5425.3 | 5425.3..5425.3 | 1 | - | - | - | 3358.9 | - | auc=0.834553, logloss=0.410556 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 2663.4 | 2663.4..2663.4 | 1 | 2.200 | 2.037 | - | 4244.9 | - | auc=0.904749, logloss=0.305763 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:127504,ours:128000,ours-ab:128000 spread=0.0039 verdict=COMPARABLE`

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
| scale_pos_weight | 11.363636363636363 | 11.363636363636363 | 11.363636363636363 |
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
| mojolearn IDENTICAL | mojolearn | gpu | identical | 3175.4 | 3175.4..3175.4 | 1 | - | - | - | 515.0 | - | auc=0.532146, logloss=0.752401 | yes | COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3458.7 | 3458.7..3458.7 | 1 | - | - | - | 592.4 | - | auc=0.532717, logloss=0.751517 | yes | COMPARABLE | wheel | ok |
| catboost-cpu | catboost | cpu | opponent | 389.7 | 389.7..389.7 | 1 | 8.147 | 8.874 | - | 658.9 | - | auc=0.548727, logloss=0.703954 | yes | COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, catboost-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=gbdt-symmetric arms=catboost-cpu,ours,ours-ab leaves=catboost-cpu:124920,ours:128000,ours-ab:128000 spread=0.0241 verdict=COMPARABLE`

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
| scale_pos_weight | 1.2987012987012987 | 1.2987012987012987 | 1.2987012987012987 |
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
| mojolearn IDENTICAL | mojolearn | gpu | identical | 88.3 | 88.3..88.3 | 1 | - | - | - | 2333.5 | - | auc=0.763032 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 88.7 | 88.7..88.7 | 1 | - | - | - | 3270.6 | - | auc=0.763032 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 48.3 | 48.3..48.3 | 1 | 1.829 | 1.839 | - | 3245.0 | - | auc=0.792792 | yes | UNKNOWN | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:5187 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

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

### iforest / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/iforest.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 84.4 | 84.4..84.4 | 1 | - | - | - | 466.7 | - | auc=0.542833 | yes | UNKNOWN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 83.4 | 83.4..83.4 | 1 | - | - | - | 532.7 | - | auc=0.542833 | yes | UNKNOWN | wheel | ok |
| sklearn-iforest-cpu | scikit-learn | cpu | opponent | 48.2 | 48.2..48.2 | 1 | 1.752 | 1.731 | - | 524.8 | - | auc=0.538780 | yes | UNKNOWN | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-iforest-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=iforest arms=ours,ours-ab,sklearn-iforest-cpu leaves=sklearn-iforest-cpu:4660 spread=- verdict=UNKNOWN reason=fewer than two arms exposed a leaf count; an unread comparison is not a fair one`

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

### rf / istella (rows 2000, shape istella-2000x220)

race: done, driver rc 0, log `raw/trees/rf.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2477.7 | 2477.7..2477.7 | 1 | - | - | - | 2656.0 | - | auc=0.885670, logloss=0.273691 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2159.3 | 2159.3..2159.3 | 1 | - | - | - | 3091.9 | - | auc=0.885670, logloss=0.273691 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 363.9 | 363.9..363.9 | 1 | 6.809 | 5.934 | - | 3234.0 | - | auc=0.886759, logloss=0.273000 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 6532.1 | 6532.1..6532.1 | 1 | 0.379 | 0.331 | - | 3209.1 | - | auc=0.884948, logloss=0.259049 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:32528,ours:29734,ours-ab:29734,sklearn-rf-cpu:29020 spread=0.1078 verdict=NOT-COMPARABLE`

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

### rf / taxi (rows 2000, shape taxi-2000x16)

race: done, driver rc 0, log `raw/trees/rf.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 2079.0 | 2079.0..2079.0 | 1 | - | - | - | 647.1 | - | auc=0.586569, logloss=0.621278 | yes | NOT-COMPARABLE | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 2001.2 | 2001.2..2001.2 | 1 | - | - | - | 691.9 | - | auc=0.586569, logloss=0.621278 | yes | NOT-COMPARABLE | wheel | ok |
| sklearn-rf-cpu | scikit-learn | cpu | opponent | 298.6 | 298.6..298.6 | 1 | 6.963 | 6.703 | - | 619.3 | - | auc=0.587152, logloss=0.593391 | yes | NOT-COMPARABLE | - | ok |
| lightgbm-cpu | lightgbm | cpu | opponent | 5097.8 | 5097.8..5097.8 | 1 | 0.408 | 0.393 | - | 612.8 | - | auc=0.579249, logloss=0.721738 | yes | NOT-COMPARABLE | - | ok |

memory, ours, ours-ab: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-rf-cpu, lightgbm-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

FSPEED-FIT-VERDICT: `lane=rf arms=lightgbm-cpu,ours,ours-ab,sklearn-rf-cpu leaves=lightgbm-cpu:26732,ours:35301,ours-ab:35301,sklearn-rf-cpu:33337 spread=0.2427 verdict=NOT-COMPARABLE`

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

## Classical

### dbscan / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.dbscan.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 99.8 | 99.8..99.8 | 1 | - | - | - | 70.1 | - | n_clusters=179, noise_fraction=0.564500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 90.7 | 90.7..90.7 | 1 | - | - | - | 69.1 | - | ari_vs_ours=1.000000, n_clusters=179, noise_agreement_vs_ours=1.000000, noise_fraction=0.564500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 30.8 | 30.8..30.8 | 1 | 3.243 | 2.949 | - | 156.1 | - | ari_vs_ours=1.000000, n_clusters=179, noise_agreement_vs_ours=1.000000, noise_fraction=0.564500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows (the full board; this run caps them at --rows 2000): dbscan block: 1,000,000 rows, standardized. Timed: fit.

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

### dbscan / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.dbscan.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 85.3 | 85.3..85.3 | 1 | - | - | - | 79.2 | - | n_clusters=10, noise_fraction=0.005500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 37.5 | 37.5..37.5 | 1 | - | - | - | 66.6 | - | ari_vs_ours=1.000000, n_clusters=10, noise_agreement_vs_ours=1.000000, noise_fraction=0.005500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 55.7 | 55.7..55.7 | 1 | 1.531 | 0.674 | - | 240.7 | - | ari_vs_ours=1.000000, n_clusters=10, noise_agreement_vs_ours=1.000000, noise_fraction=0.005500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: eps=3, min_samples=2 (the cuML benchmark's DBSCAN) on every arm; metric='euclidean'. Rows (the full board; this run caps them at --rows 2000): dbscan block: 1,000,000 rows, standardized. Timed: fit.

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

### hdbscan / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.hdbscan.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 144.1 | 144.1..144.1 | 1 | - | - | - | 92.1 | - | n_clusters=5, noise_fraction=0.492500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 120.9 | 120.9..120.9 | 1 | - | - | - | 85.0 | - | ari_vs_ours=1.000000, n_clusters=5, noise_agreement_vs_ours=1.000000, noise_fraction=0.492500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 425.9 | 425.9..425.9 | 1 | 0.338 | 0.284 | - | 147.7 | - | ari_vs_ours=0.931372, n_clusters=5, noise_agreement_vs_ours=0.972500, noise_fraction=0.468000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows (the full board; this run caps them at --rows 2000): the dbscan block's first 100,000 rows. Timed: fit.

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

### hdbscan / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.hdbscan.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 67.0 | 67.0..67.0 | 1 | - | - | - | 78.0 | - | n_clusters=4, noise_fraction=0.211500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 79.4 | 79.4..79.4 | 1 | - | - | - | 75.7 | - | ari_vs_ours=1.000000, n_clusters=4, noise_agreement_vs_ours=1.000000, noise_fraction=0.211500, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 94.6 | 94.6..94.6 | 1 | 0.708 | 0.840 | - | 138.2 | - | ari_vs_ours=0.991083, n_clusters=4, noise_agreement_vs_ours=0.994500, noise_fraction=0.206000, rows=2000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: min_samples=10, min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False. Rows (the full board; this run caps them at --rows 2000): the dbscan block's first 100,000 rows. Timed: fit.

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

### kde / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.kde.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.0 | 14.0..14.0 | 1 | - | - | - | 63.4 | - | mean_log_likelihood=-398.822047, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 62.4 | 62.4..62.4 | 1 | - | - | - | 60.5 | - | mean_log_likelihood=-398.822031, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 65.4 | 65.4..65.4 | 1 | 0.215 | 0.954 | - | 142.6 | - | mean_log_likelihood=-397.566925, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows (the full board; this run caps them at --rows 2000): kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

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

### kde / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.kde.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.1 | 7.1..7.1 | 1 | - | - | - | 57.1 | - | mean_log_likelihood=-14.818538, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 6.3 | 6.3..6.3 | 1 | - | - | - | 55.6 | - | mean_log_likelihood=-14.818538, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 13.2 | 13.2..13.2 | 1 | 0.541 | 0.476 | - | 133.9 | - | mean_log_likelihood=-14.818536, rows_without_density=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: bandwidth=1.0, kernel='gaussian' (the cuML benchmark's KernelDensity), metric='euclidean' on every arm; ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, breadth_first=True. Rows (the full board; this run caps them at --rows 2000): kde block: 100,000 fit rows, 2,000 queries, standardized. Timed: score_samples; the fit is before the clock on every arm.

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

### kmeans / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.kmeans.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 108.7 | 108.7..108.7 | 1 | - | - | - | 78.2 | - | inertia=4.136e+14, inertia_over_ours=1.000000, n_iter=8 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 26.9 | 26.9..26.9 | 1 | - | - | - | 68.7 | - | inertia=4.136e+14, inertia_over_ours=1.000000, n_iter=8 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 33.5 | 33.5..33.5 | 1 | 3.244 | 0.802 | - | 144.1 | - | inertia=4.115e+14, inertia_over_ours=0.994966, n_iter=5 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 47.0 | 47.0..47.0 | 1 | 2.311 | 0.571 | - | 224.0 | 50.6 | inertia=4.117e+14, inertia_over_ours=0.995480, n_iter=6 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows (the full board; this run caps them at --rows 2000): big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

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

### kmeans / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.kmeans.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 20.0 | 20.0..20.0 | 1 | - | - | - | 65.9 | - | inertia=25888.434366, inertia_over_ours=1.000000, n_iter=14 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 60.8 | 60.8..60.8 | 1 | - | - | - | 64.4 | - | inertia=25888.434366, inertia_over_ours=1.000000, n_iter=14 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 26.8 | 26.8..26.8 | 1 | 0.745 | 2.266 | - | 140.0 | - | inertia=27617.726175, inertia_over_ours=1.066798, n_iter=67 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 85.2 | 85.2..85.2 | 1 | 0.234 | 0.713 | - | 190.7 | 18.6 | inertia=27982.170005, inertia_over_ours=1.080875, n_iter=26 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_clusters=8, init='k-means++', max_iter=300, n_init=1 (the cuML benchmark's KMeans), oversampling_factor=0 on ours and cuML (its cuml_args; the classic sequential seeding), tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn and cuML random_state=7, torch a generator seeded 7). Rows (the full board; this run caps them at --rows 2000): big block: 4,000,000 taxi rows or the Istella-S train split, raw. Timed: fit (the k-means++ seeding included on every arm).

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

### knn / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.knn.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 15.3 | 15.3..15.3 | 1 | - | - | - | 61.9 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 11.3 | 11.3..11.3 | 1 | - | - | - | 60.2 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 10.8 | 10.8..10.8 | 1 | 1.420 | 1.046 | - | 139.3 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 13.1 | 13.1..13.1 | 1 | 1.168 | 0.860 | - | 205.5 | 40.5 | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows (the full board; this run caps them at --rows 2000): knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

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

### knn / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.knn.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 14.9 | 14.9..14.9 | 1 | - | - | - | 57.0 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 9.1 | 9.1..9.1 | 1 | - | - | - | 57.9 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7.9 | 7.9..7.9 | 1 | 1.878 | 1.149 | - | 140.2 | - | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | 12.0 | 12.0..12.0 | 1 | 1.239 | 0.758 | - | 173.0 | 8.5 | recall_at_k=1.000000, rows_with_repeated_ids=0 | yes | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU torch.mps.driver_allocated_memory at the round's end (not a peak; unified memory, also inside peak_host_mb)

settings: n_neighbors=64 (the cuML benchmark's NearestNeighbors), metric='euclidean', algorithm='brute' (ours, scikit-learn, cuML); torch cdist p=2 plus topk. Rows (the full board; this run caps them at --rows 2000): knn block: 400,000 index rows, 4,000 queries, raw. Timed: kneighbors; the fit (index) is before the clock on every arm.

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

### ols / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.ols.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 1033.7 | 1033.7..1033.7 | 1 | - | - | - | 67.4 | - | finite=True, r2=-24.129541, rmse=4.746592 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 949.4 | 949.4..949.4 | 1 | - | - | - | 66.7 | - | finite=True, r2=-24.390212, rmse=4.771147 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 16.0 | 16.0..16.0 | 1 | 64.410 | 59.160 | - | 146.2 | - | finite=True, r2=-0.027117, rmse=0.959621 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: fit_intercept=True. Rows (the full board; this run caps them at --rows 2000): big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

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

### ols / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.ols.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 9.6 | 9.6..9.6 | 1 | - | - | - | 56.9 | - | finite=True, r2=0.889788, rmse=6.579774 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 4.9 | 4.9..4.9 | 1 | - | - | - | 57.9 | - | finite=True, r2=0.889542, rmse=6.587113 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 2.2 | 2.2..2.2 | 1 | 4.403 | 2.263 | - | 139.2 | - | finite=True, r2=0.907724, rmse=6.020627 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::linalg_lstsq.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://gith) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: fit_intercept=True. Rows (the full board; this run caps them at --rows 2000): big block, raw; R2 and RMSE on the 500,000 eval rows. Timed: fit.

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

### pca / istella (rows 2000, shape 2000x220)

race: done, driver rc 0, log `logs/classical.pca.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 182.6 | 182.6..182.6 | 1 | - | - | - | 66.4 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 158.7 | 158.7..158.7 | 1 | - | - | - | 66.1 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 8.3 | 8.3..8.3 | 1 | 22.028 | 19.145 | - | 144.4 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::_linalg_eigh.eigenvalues' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on http) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows (the full board; this run caps them at --rows 2000): big block, raw. Timed: fit.

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

### pca / taxi (rows 2000, shape 2000x11)

race: done, driver rc 0, log `logs/classical.pca.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 7.6 | 7.6..7.6 | 1 | - | - | - | 56.7 | - | explained_variance_ratio_sum=1.000001 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 3.6 | 3.6..3.6 | 1 | - | - | - | 55.5 | - | explained_variance_ratio_sum=1.000001 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 3.2 | 3.2..3.2 | 1 | 2.390 | 1.124 | - | 138.5 | - | explained_variance_ratio_sum=1.000000 | yes | LIKE-FOR-LIKE-SPAN | - | ok |
| torch-gpu | torch | gpu | opponent | - | - | 0 | - | - | - | - | - | - | - | SPAN-ASYMMETRIC(upload_outside_its_clock) | - | REFUSED(error: {"error": "NotImplementedError(\"The operator 'aten::_linalg_eigh.eigenvalues' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on http) |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

memory, torch-gpu: host not sampled; GPU not sampled

settings: n_components=10 (the cuML benchmark's PCA), whiten=False, random_state=7; ours and scikit-learn svd_solver='covariance_eigh'. Rows (the full board; this run caps them at --rows 2000): big block, raw. Timed: fit.

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

### svc / istella (rows 2000, shape 256x220)

race: done, driver rc 0, log `logs/classical.svc.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 27.4 | 27.4..27.4 | 1 | - | - | - | 69.4 | - | accuracy=0.855469, n_support=94 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 27.6 | 27.6..27.6 | 1 | - | - | - | 65.4 | - | accuracy=0.855469, n_support=94 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 7.8 | 7.8..7.8 | 1 | 3.497 | 3.522 | - | 137.9 | - | accuracy=0.855469, n_support=94 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows (the full board; this run caps them at --rows 2000): svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

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

### svc / taxi (rows 2000, shape 256x11)

race: done, driver rc 0, log `logs/classical.svc.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 34.9 | 34.9..34.9 | 1 | - | - | - | 64.0 | - | accuracy=0.765625, n_support=151 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 28.9 | 28.9..28.9 | 1 | - | - | - | 64.4 | - | accuracy=0.765625, n_support=151 | yes | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 6.4 | 6.4..6.4 | 1 | 5.488 | 4.547 | - | 136.8 | - | accuracy=0.765625, n_support=151 | yes | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, cache_size=2000 MB, class_weight=None. Rows (the full board; this run caps them at --rows 2000): svc block: 10,000 fit rows, 10,000 eval rows, standardized. Timed: fit.

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

## Classical, wave 2

### gmm / istella (rows 2000, shape X 2000x220; Xq 2000x220)

race: failed, driver rc 1, log `logs/classical2.gmm.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"GaussianMixture: fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to d) |
| mojolearn FAST | mojolearn | gpu | fast | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | wheel | REFUSED(error: {"error": "Exception(\"GaussianMixture: fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to d) |
| sklearn-cpu | scikit-learn | cpu | opponent | - | - | 0 | - | - | - | - | - | - | - | LIKE-FOR-LIKE-SPAN | - | REFUSED(error: {"error": "ValueError('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the numbe) |

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

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

### gmm / taxi (rows 2000, shape X 2000x11; Xq 2000x11)

race: done, driver rc 0, log `logs/classical2.gmm.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 244.2 | 244.2..244.2 | 1 | - | - | - | 80.5 | - | bic=-68064.910372, mean_log_likelihood=12.918593, n_iter=25 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 83.2 | 83.2..83.2 | 1 | - | - | - | 63.8 | - | bic=-65151.291133, mean_log_likelihood=12.570116, n_iter=11 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 154.0 | 154.0..154.0 | 1 | 1.586 | 0.540 | - | 138.3 | - | bic=-69061.429868, mean_log_likelihood=13.245844, n_iter=100 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, init_params='kmeans', n_init=1, warm_start=False, random_state=7. Rows (the full board; this run caps them at --rows 2000): 100000 fit and 20000 held-out stride rows of the reg block (standardized by the fit rows). Timed: fit.

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

### linearsvc / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.linearsvc.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 137.0 | 137.0..137.0 | 1 | - | - | - | 64.8 | - | accuracy=0.913500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 141.4 | 141.4..141.4 | 1 | - | - | - | 65.5 | - | accuracy=0.912500 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 929.5 | 929.5..929.5 | 1 | 0.147 | 0.152 | - | 140.3 | - | accuracy=0.913500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

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

### linearsvc / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.linearsvc.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 8.8 | 8.8..8.8 | 1 | - | - | - | 59.7 | - | accuracy=0.763000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 10.4 | 10.4..10.4 | 1 | - | - | - | 58.3 | - | accuracy=0.763000 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.7 | 1.7..1.7 | 1 | 5.157 | 6.061 | - | 135.7 | - | accuracy=0.762500 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; ours and cuML penalized_intercept=False; scikit-learn intercept_scaling=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

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

### logreg / istella (rows 2000, shape X 2000x220; Xq 2000x220; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.logreg.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 136.9 | 136.9..136.9 | 1 | - | - | - | 63.7 | - | accuracy=0.923500, logloss=0.204798, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 133.8 | 133.8..133.8 | 1 | - | - | - | 64.3 | - | accuracy=0.923000, logloss=0.204828, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 18.9 | 18.9..18.9 | 1 | 7.255 | 7.094 | - | 138.4 | - | accuracy=0.923500, logloss=0.204942, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

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

### logreg / taxi (rows 2000, shape X 2000x11; Xq 2000x11; y 2000; yq 2000)

race: done, driver rc 0, log `logs/classical2.logreg.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 12.5 | 12.5..12.5 | 1 | - | - | - | 58.5 | - | accuracy=0.763500, logloss=0.547874, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 16.8 | 16.8..16.8 | 1 | - | - | - | 58.6 | - | accuracy=0.763500, logloss=0.547874, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 1.9 | 1.9..1.9 | 1 | 6.557 | 8.844 | - | 132.3 | - | accuracy=0.763500, logloss=0.547867, nonfinite_proba_rows=0 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, class_weight=None; scikit-learn random_state=7. Rows (the full board; this run caps them at --rows 2000): 1000000 fit and 100000 held-out stride rows (standardized by the fit rows). Timed: fit.

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

### spectral-embedding / istella (rows 2000, shape X 2000x220)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 58.8 | 58.8..58.8 | 1 | - | - | - | 100.4 | - | trustworthiness_k15=0.858681 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 63.2 | 63.2..63.2 | 1 | - | - | - | 97.1 | - | trustworthiness_k15=0.858681 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 94.6 | 94.6..94.6 | 1 | 0.621 | 0.668 | - | 156.5 | - | trustworthiness_k15=0.858691 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows (the full board; this run caps them at --rows 2000): the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

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

### spectral-embedding / taxi (rows 2000, shape X 2000x11)

race: done, driver rc 0, log `logs/classical2.spectral-embedding.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 54.5 | 54.5..54.5 | 1 | - | - | - | 96.1 | - | trustworthiness_k15=0.690399 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 46.7 | 46.7..46.7 | 1 | - | - | - | 89.3 | - | trustworthiness_k15=0.708797 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| sklearn-cpu | scikit-learn | cpu | opponent | 144.8 | 144.8..144.8 | 1 | 0.377 | 0.322 | - | 139.1 | - | trustworthiness_k15=0.687460 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, sklearn-cpu: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7. Rows (the full board; this run caps them at --rows 2000): the umap block (20000 stride rows, standardized by the fit rows). Timed: fit_transform.

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

### umap / istella (rows 2000, shape X 2000x220)

race: done, driver rc 0, log `logs/classical2.umap.istella.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 149.5 | 149.5..149.5 | 1 | - | - | - | 100.6 | - | trustworthiness_k15=0.943567 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 227.6 | 227.6..227.6 | 1 | - | - | - | 95.2 | - | trustworthiness_k15=0.942379 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 2350.4 | 2350.4..2350.4 | 1 | 0.064 | 0.097 | - | 403.5 | - | trustworthiness_k15=0.945192 | - | LIKE-FOR-LIKE-SPAN | - | ok |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1453.3 | 1453.3..1453.3 | 1 | 0.103 | 0.157 | - | 408.5 | - | trustworthiness_k15=0.942601 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

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

### umap / taxi (rows 2000, shape X 2000x11)

race: done, driver rc 0, log `logs/classical2.umap.taxi.rows-2000.log`

| arm | library | device | mode | median ms | min..max ms | rounds | ours IDENTICAL / arm | ours FAST / arm | ours CPU / arm | peak host MB | peak GPU MB | quality | hash stable | comparability | installed_wheel | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mojolearn IDENTICAL | mojolearn | gpu | identical | 165.4 | 165.4..165.4 | 1 | - | - | - | 91.6 | - | trustworthiness_k15=0.973542 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| mojolearn FAST | mojolearn | gpu | fast | 219.8 | 219.8..219.8 | 1 | - | - | - | 90.4 | - | trustworthiness_k15=0.973370 | - | LIKE-FOR-LIKE-SPAN | wheel | ok |
| umap-learn-cpu | umap-learn | cpu | opponent | 2212.0 | 2212.0..2212.0 | 1 | 0.075 | 0.099 | - | 454.6 | - | trustworthiness_k15=0.973861 | - | LIKE-FOR-LIKE-SPAN | - | ok |
| umap-learn-cpu-unseeded | umap-learn | cpu | opponent | 1393.2 | 1393.2..1393.2 | 1 | 0.119 | 0.158 | - | 460.0 | - | trustworthiness_k15=0.975341 | - | LIKE-FOR-LIKE-SPAN | - | ok |

memory, ours, ours-fast: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU Apple unified memory: no per-process GPU counter; Metal buffers are inside peak_host_mb (phys_footprint)

memory, umap-learn-cpu, umap-learn-cpu-unseeded: host macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical footprint over the round; Metal buffers are inside it); GPU cpu arm: no device memory

settings: n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, min_dist=0.1, spread=1.0, metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7. Rows (the full board; this run caps them at --rows 2000): 20000 stride rows of the train split, standardized by the fit rows. Timed: fit (ours fit_transform) from host rows to the embedding.

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

