# M3 FAST board refresh (lane/apple-fast)

Our FAST arm on the M3 Ultra Metal GPU at head 05fffc97d, 0e743cac9, 24ed76679, 2dcdd949f, 37c65a3af, 413da4d18, 417ba1ded, 4198d5a9c, 42b6db46c, 4b1311c12, 550806bc0, 5990c5946, 69f7a41fd, 6c0ba4379, 6c54b7e87, 6d4d55c99, 73f3a856d, 829c3fb4a, 8b9c91b9c, 8d43ec357, 94cb5ba59, 9525ed958, 9722e5a2b, a7b8b9513, a8e9ee5a3, b2dd5dfe5, c48ede3c1, c55b8c377, c5e1bbeb6, c73305a4c, d1fa9223b, deee07721, e2bfb8422, eb3fca1ac, ed49f2b11, f419ea9f1, f6a9e7c04, fc6c6837f, 1 warm-up + 3 timed rounds at board size (rows-full; trees MOJOLEARN_SPEED_SIZE=shipped). Opponents are not re-raced: classical times come from the M3 0.8.34 board (`~/mojolearn-evidence/board-0834-times.tsv`, quality from its board.json), trees from the 2026-09-29 M3 board (older tree params on some lanes); an opponent marked (fill) comes from the M3 opponent fill on the 0.8.34 kit. Ratio = our FAST ms / best opponent ms; below 1 is faster. Rows sort worst ratio after first. Written by `tools/af_board_merge.py`.

Summary: 243 rows, 231 with a ratio, 185 faster than the best opponent after (FAST quality under review, not counted: factor-analysis istella, tweedie istella), geometric-mean ratio 0.34. Flips to faster: lasso-lars istella, onehot istella, ordinal istella, multioutput-reg taxi, qda taxi, minibatch-kmeans taxi, iforest istella, complement-nb text, bisecting-kmeans taxi, kpss synthetic, connected-components istella, dart-reg istella, gpr taxi, connected-components taxi, gaussian-rp istella, dart istella, maxabs-scaler taxi, knn istella, label-encoder istella, select-d taxi-hourly, kpss taxi-hourly, minmax-scaler taxi, lr-onecycle synthetic, nearest-centroid istella, enet-cv taxi, meanshift istella, label-encoder taxi, select-f-regression taxi, ard taxi, gbdt-rank-yetirank istella, gbdt-categorical taxi, lasso-cv taxi, ordinal taxi, isotonic taxi, select-f-classif taxi, select-d synthetic, knn-clf istella, lasso-lars taxi, knn-reg istella, ocsvm taxi, lars taxi, elasticnet taxi, gbdt-lossguide istella, select-r-regression taxi, ridge-clf istella, minmax-scaler istella, onehot taxi, lstsq istella, lasso taxi, ridge-clf taxi, categorical-nb taxi, nearest-centroid taxi, bayesian-ridge taxi, categorical-nb istella, lda-clf istella, theta taxi-hourly, huber taxi, lasso-cv istella, damped-ets synthetic, damped-ets taxi-hourly, calibrated taxi, enet-cv istella, ard istella, bayesian-ridge istella, lda taxi-zones, stl synthetic, qda istella, stl taxi-hourly, huber istella. Flips to slower: adafactor synthetic.

| lane | dataset | family | FAST before ms | FAST after ms | best opponent | opp ms | ratio before | ratio after | flip | quality after (FAST) | quality before (FAST) | opponent quality | status |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| elliptic-envelope | taxi | algos | 1741 | 78766 | sklearn-cpu | 1423 | 1.22 | 55.34 |  | fraction_flagged=0.10237 | fraction_flagged=0.1024 | fraction_flagged=0.1025, jaccard_vs_sklearn=1 | ok |
| layernorm | synthetic | algos | 26.8 | 52.2 | torch-eager-bf16 | 2.8 | 9.57 | 18.64 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| adagrad | synthetic | algos | 131 | 196 | torch-eager-fp32 | 14.2 | 9.25 | 13.81 |  | - | - | - | ok |
| adam | synthetic | algos | 125 | 343 | torch-eager-fp32 | 29.3 | 4.27 | 11.69 |  | - | - | - | ok |
| adamw | synthetic | algos | 114 | 338 | torch-eager-fp32 | 31.1 | 3.66 | 10.88 |  | - | - | - | ok |
| autoarima | taxi-hourly | algos | - | 24563 | statsforecast-cpu | 2844 | - | 8.64 |  | forecast_rmse=74.6591 | - | forecast_rmse=68.21 | ok |
| rmsprop | synthetic | algos | 129 | 195 | torch-eager-fp32 | 25.2 | 5.13 | 7.74 |  | - | - | - | ok |
| adamax | synthetic | algos | 176 | 205 | torch-eager-fp32 | 27.7 | 6.37 | 7.39 |  | - | - | - | ok |
| nadam | synthetic | algos | 178 | 204 | torch-eager-fp32 | 32.9 | 5.40 | 6.20 |  | - | - | - | ok |
| autoarima | synthetic | algos | - | 14918 | statsforecast-cpu | 2880 | - | 5.18 |  | forecast_rmse=2.62412 | - | forecast_rmse=17.55 | ok |
| label-binarizer | taxi | algos | 1362 | 562 | sklearn-cpu | 118 | 11.59 | 4.78 |  | output_shape=100000x259 | - | - | ok |
| cholesky | synthetic | algos | 960 | 431 | torch-gpu | 110 | 8.70 | 3.91 |  | relative_residual=1.6591e-07 | relative_residual=1.659e-07 | relative_residual=5.449e-07 | ok |
| kernel-shap | istella | algos | 14386 | 27005 | shap-cpu | 7696 | 1.87 | 3.51 |  | rel_error_vs_exact=4.37806e-09 | rel_error_vs_exact=4.179e-09 | rel_error_vs_exact=1.405e-14 | ok |
| additive-chi2 | istella | algos | 11.0 | 12.3 | sklearn-cpu | 3.8 | 2.89 | 3.23 |  | kernel_rel_error=0.0877304 | kernel_rel_error=0.08773 | kernel_rel_error=0.08773 | ok |
| lu-factor | synthetic | algos | 2323 | 1367 | scipy-cpu | 428 | 5.43 | 3.19 |  | relative_residual=3.2563e-06 | relative_residual=3.248e-06 | relative_residual=3.246e-06 | ok |
| lle | taxi | algos | 17885 | 3834 | sklearn-cpu | 1247 | 14.35 | 3.07 |  | trustworthiness_k15=0.826083 | trustworthiness_k15=0.8398 | trustworthiness_k15=0.7708 | ok |
| lr-exponential | synthetic | algos | 224 | 225 | torch-cpu | 76.7 | 2.92 | 2.93 |  | - | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.933e-08 | ok |
| cagra | taxi | algos | - | 2900 | faiss-cpu | 1016 | - | 2.86 |  | recall_at_10=0.997925 | - | recall_at_10=0.9277 | ok |
| lu-solve | synthetic | algos | 2320 | 1378 | torch-gpu | 506 | 4.59 | 2.73 |  | relative_residual=3.2563e-06 | relative_residual=3.248e-06 | relative_residual=8.234e-07 | ok |
| lstm-reg | synthetic | algos | 1880 | 1878 | torch-compile-fp32 (fill) | 695 | 2.70 | 2.70 |  | r2=0.981013, rmse=0.159641 | r2=0.981, rmse=0.1596 | - | ok |
| lstm-reg | taxi-hourly | algos | 1878 | 1879 | torch-compile-fp32 (fill) | 700 | 2.68 | 2.68 |  | r2=0.751679, rmse=0.540429 | r2=0.7517, rmse=0.5404 | - | ok |
| lstm-clf | taxi-hourly | algos | 1879 | 1890 | torch-eager-fp32 (fill) | 729 | 2.58 | 2.59 |  | accuracy=0.868218, logloss=0.299901 | accuracy=0.8682, logloss=0.2999 | - | ok |
| lstm-clf | synthetic | algos | 1878 | 1882 | torch-eager-fp32 (fill) | 729 | 2.58 | 2.58 |  | accuracy=0.968696, logloss=0.0724409 | accuracy=0.9687, logloss=0.07244 | - | ok |
| svgp | taxi | algos | - | 609 | gpytorch-cpu | 248 | - | 2.46 |  | r2=-0.194982, rmse=17.7235 | - | r2=-0.2093, rmse=17.83 | ok |
| gaussian-rp | taxi | algos | 5.9 | 3.9 | sklearn-cpu | 1.6 | 3.69 | 2.44 |  | mean_abs_distortion=0.345752 | mean_abs_distortion=0.3458 | mean_abs_distortion=0.3398 | ok |
| kernel-pca | taxi | algos | 847 | 945 | sklearn-cpu | 460 | 1.84 | 2.05 |  | - | - | subspace_cos_vs_sklearn=1 | ok |
| var | synthetic | algos | 15.9 | 5.0 | statsmodels-cpu | 2.6 | 6.12 | 1.93 |  | forecast_rmse=1.14079 | forecast_rmse=1.141 | forecast_rmse=1.145 | ok |
| moe | synthetic | algos | 789 | 72.7 | torch-eager-bf16 | 37.7 | 20.92 | 1.93 |  | - | - | max_rel_diff_vs_torch_eager_fp32=2.234e+04, rel_fro_vs_torch_eager_fp32=0.05522 | ok |
| kernel-pca | istella | algos | 1049 | 1038 | sklearn-cpu | 560 | 1.87 | 1.85 |  | - | - | subspace_cos_vs_sklearn=1 | ok |
| var | taxi-hourly | algos | 14.5 | 5.1 | statsmodels-cpu | 2.8 | 5.18 | 1.82 |  | forecast_rmse=33.168 | forecast_rmse=33.17 | forecast_rmse=33.17 | ok |
| additive-chi2 | taxi | algos | 0.6 | 0.7 | sklearn-cpu | 0.4 | 1.50 | 1.69 |  | kernel_rel_error=0.0938923 | kernel_rel_error=0.09389 | kernel_rel_error=0.09389 | ok |
| bisecting-kmeans | istella | algos | 3908 | 1994 | sklearn-cpu | 1428 | 2.74 | 1.40 |  | n_clusters=8, silhouette=0.118345 | ari_vs_ours=1, n_clusters=8, silhouette=0.1183 | ari_vs_ours=0.6929, n_clusters=8, silhouette=0.09641 | ok |
| minibatch-kmeans | istella | algos | 673 | 171 | sklearn-cpu | 124 | 5.45 | 1.38 |  | n_clusters=8, silhouette=0.118168 | ari_vs_ours=1, n_clusters=8, silhouette=0.1167 | ari_vs_ours=0.6224, n_clusters=8, silhouette=0.1119 | ok |
| isotonic | istella | algos | 608 | 51.3 | sklearn-cpu | 37.5 | 16.22 | 1.37 |  | r2=0.187985, rmse=0.752735 | r2=0.188, rmse=0.7527 | r2=0.188, rmse=0.7527 | ok |
| lle | istella | algos | 17825 | 3899 | sklearn-cpu | 2852 | 6.25 | 1.37 |  | trustworthiness_k15=0.895277 | trustworthiness_k15=0.8724 | trustworthiness_k15=0.8491 | ok |
| label-binarizer | istella | algos | 326 | 43.2 | sklearn-cpu | 31.7 | 10.27 | 1.36 |  | output_shape=100000x16 | - | - | ok |
| knn-imputer | taxi | algos | 73.6 | 2.3 | sklearn-cpu | 1.7 | 43.29 | 1.36 |  | masked_rmse=6.1517 | masked_rmse=6.152 | masked_rmse=5.257 | ok |
| mb-dict-learning | istella | algos | 7966 | 6841 | sklearn-cpu | 5030 | 1.58 | 1.36 |  | component_sparsity=0.0863636, relative_reconstruction_error=0.64834 | component_sparsity=0.08636, relative_reconstruction_error=0.6483 | component_sparsity=0.08636, relative_reconstruction_error=0.644 | ok |
| resample | taxi | algos | 65.2 | 68.1 | sklearn-cpu | 51.9 | 1.26 | 1.31 |  | max_mean_shift_over_std=0.0029166 | max_mean_shift_over_std=0.002917 | max_mean_shift_over_std=0.002257 | ok |
| maxabs-scaler | istella | algos | 134 | 124 | sklearn-cpu | 99.7 | 1.34 | 1.24 |  | output_shape=100000x220 | - | - | ok |
| resample | istella | algos | 388 | 394 | sklearn-cpu | 336 | 1.16 | 1.17 |  | max_mean_shift_over_std=0.00320274 | max_mean_shift_over_std=0.003203 | max_mean_shift_over_std=0.002552 | ok |
| adafactor | synthetic | algos | 184 | 393 | torch-eager-fp32 | 344 | 0.54 | 1.15 | FLIP slower | - | - | - | ok |
| incremental-pca | taxi | algos | 483 | 158 | sklearn-cpu | 138 | 3.50 | 1.14 |  | explained_variance_fraction=0.999995 | explained_variance_fraction=1 | explained_variance_fraction=1 | ok |
| gbdt-depthwise | taxi | trees | 13994 | 10812 | xgboost-cpu (fill) | 10435 | 1.34 | 1.04 |  | logloss=0.527929, auc=0.632351 | auc=0.6258, logloss=0.53 | auc=0.631 | ok |
| lasso-lars | istella | algos | 1016 | 227 | sklearn-cpu | 228 | 4.45 | 1.00 | FLIP faster | r2=0.310837, rmse=0.69346 | r2=0.3103, rmse=0.6937 | r2=0.3111, rmse=0.6933 | ok |
| onehot | istella | algos | 78.8 | 36.5 | sklearn-cpu | 36.7 | 2.15 | 0.99 | FLIP faster | - | - | - | ok |
| ordinal | istella | algos | 79.8 | 36.2 | sklearn-cpu | 36.9 | 2.16 | 0.98 | FLIP faster | - | - | - | ok |
| multioutput-reg | taxi | algos | 115 | 50.2 | sklearn-cpu | 53.7 | 2.14 | 0.94 | FLIP faster | r2=0.60424 | r2=0.6042 | r2=0.6043 | ok |
| qda | taxi | algos | 187 | 123 | sklearn-cpu | 141 | 1.33 | 0.87 | FLIP faster | accuracy=0.72722, logloss=1.05927 | accuracy=0.727, logloss=1.061 | accuracy=0.7272, logloss=1.059 | ok |
| minibatch-kmeans | taxi | algos | 106 | 38.1 | sklearn-cpu | 43.9 | 2.40 | 0.87 | FLIP faster | n_clusters=8, silhouette=0.138023 | ari_vs_ours=1, n_clusters=8, silhouette=0.1381 | ari_vs_ours=0.5252, n_clusters=8, silhouette=0.1655 | ok |
| iforest | istella | trees | 437 | 262 | sklearn-iforest-cpu (fill) | 304 | 1.44 | 0.86 | FLIP faster | auc=0.830358 | auc=0.8304 | auc=0.8279 | ok |
| mlp-reg | taxi | algos | 8033 | 8031 | sklearn-cpu | 9959 | 0.81 | 0.81 |  | r2=0.931981, rmse=4.15387 | r2=0.932, rmse=4.154 | r2=0.9296, rmse=4.226 | ok |
| mlp-reg | istella | algos | 10761 | 10761 | sklearn-cpu | 13651 | 0.79 | 0.79 |  | r2=0.526405, rmse=0.574862 | r2=0.5273, rmse=0.5743 | r2=0.5248, rmse=0.5758 | ok |
| complement-nb | text | algos | 281 | 206 | sklearn-cpu | 265 | 1.06 | 0.78 | FLIP faster | accuracy=0.983067, logloss=0.559491 | accuracy=0.9831, logloss=0.5595 | accuracy=0.9831, logloss=0.5573 | ok |
| bisecting-kmeans | taxi | algos | 396 | 296 | sklearn-cpu | 383 | 1.03 | 0.77 | FLIP faster | n_clusters=8, silhouette=0.155357 | ari_vs_ours=1, n_clusters=8, silhouette=0.1554 | ari_vs_ours=0.5023, n_clusters=8, silhouette=0.128 | ok |
| cagra | istella | algos | - | 1254 | faiss-cpu | 1634 | - | 0.77 |  | recall_at_10=0.997225 | - | recall_at_10=0.9994 | ok |
| rnn-reg | synthetic | algos | 1497 | 1564 | torch-eager-fp32 (fill) | 2037 | 0.73 | 0.77 |  | r2=0.977348, rmse=0.174374 | r2=0.9774, rmse=0.1744 | - | ok |
| kpss | synthetic | algos | 3.1 | 2.0 | statsmodels-cpu | 2.6 | 1.19 | 0.76 | FLIP faster | stationary_fraction=0.03125 | stationary_fraction=0.03125 | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| rnn-reg | taxi-hourly | algos | 1572 | 1560 | torch-eager-fp32 (fill) | 2049 | 0.77 | 0.76 |  | r2=0.738796, rmse=0.554271 | r2=0.7388, rmse=0.5543 | - | ok |
| connected-components | istella | algos | 93.4 | 5.3 | networkx-cpu | 7.0 | 13.34 | 0.76 | FLIP faster | n_components=81 | n_components=81 | n_components=81 | ok |
| mlp-clf | istella | algos | 10766 | 10773 | sklearn-cpu | 14327 | 0.75 | 0.75 |  | accuracy=0.94435, logloss=0.136481 | accuracy=0.9446, logloss=0.1364 | accuracy=0.9438, logloss=0.1368 | ok |
| lof | istella | algos | - | 7580 | sklearn-cpu | 10177 | - | 0.74 |  | fraction_flagged=0.03361 | - | fraction_flagged=0.03361, jaccard_vs_sklearn=1 | ok |
| gmm | istella | classical2 | 6574 | 5858 | sklearn-cpu | 7866 | 0.84 | 0.74 |  | bic=-3.85139e+07, mean_log_likelihood=200.794, n_iter=24 | bic=-3.851e+07, mean_log_likelihood=200.8, n_iter=24 | bic=-3.901e+07, mean_log_likelihood=200.8, n_iter=30 | ok |
| label-spreading | istella | algos | - | 7508 | sklearn-cpu | 10273 | - | 0.73 |  | accuracy=0.90445 | - | accuracy=0.9044 | ok |
| auto-theta | taxi-hourly | algos | 3433 | 2910 | statsforecast-cpu | 4032 | 0.85 | 0.72 |  | forecast_rmse=49.0542 | forecast_rmse=49.31 | forecast_rmse=49.27 | ok |
| dart-reg | istella | algos | 69437 | 23099 | lightgbm-cpu | 32389 | 2.14 | 0.71 | FLIP faster | r2=0.551355, rmse=0.559515 | r2=0.5507, rmse=0.5599 | r2=0.5647, rmse=0.5512 | ok |
| gbdt-depthwise | istella | trees | 19062 | 16551 | xgboost-cpu (fill) | 23218 | 0.82 | 0.71 |  | logloss=0.156903, auc=0.98319 | auc=0.9802, logloss=0.1819 | auc=0.9836 | ok |
| rnn-clf | taxi-hourly | algos | 1485 | 1484 | torch-eager-fp32 (fill) | 2087 | 0.71 | 0.71 |  | accuracy=0.868056, logloss=0.304864 | accuracy=0.8681, logloss=0.3049 | - | ok |
| rnn-clf | synthetic | algos | 1568 | 1483 | torch-eager-fp32 (fill) | 2088 | 0.75 | 0.71 |  | accuracy=0.953559, logloss=0.103698 | accuracy=0.9536, logloss=0.1037 | - | ok |
| gpr | taxi | classical2 | 206 | 116 | sklearn-cpu | 164 | 1.26 | 0.71 | FLIP faster | mean_log_predictive_density=-311.454, r2=0.88963, rmse=5.04164 | mean_log_predictive_density=-311.4, r2=0.8896, rmse=5.042 | mean_log_predictive_density=-311.5, r2=0.8896, rmse=5.042 | ok |
| mlp-clf | taxi | algos | 8052 | 8050 | sklearn-cpu | 11431 | 0.70 | 0.70 |  | accuracy=0.76783, logloss=0.530453 | accuracy=0.7678, logloss=0.5305 | accuracy=0.7678, logloss=0.5304 | ok |
| theta | synthetic | algos | 99.6 | 72.0 | statsmodels-cpu | 104 | 0.96 | 0.69 |  | forecast_rmse=1.43661 | forecast_rmse=1.437 | forecast_rmse=1.435 | ok |
| connected-components | taxi | algos | 88.1 | 4.9 | networkx-cpu | 7.3 | 12.07 | 0.68 | FLIP faster | n_components=588 | n_components=588 | n_components=588 | ok |
| gaussian-rp | istella | algos | 58.8 | 15.9 | sklearn-cpu | 24.3 | 2.42 | 0.65 | FLIP faster | mean_abs_distortion=0.680693 | mean_abs_distortion=0.6807 | mean_abs_distortion=0.178 | ok |
| ocsvm | istella | algos | 446 | 451 | sklearn-cpu | 691 | 0.65 | 0.65 |  | fraction_flagged=0.0783 | fraction_flagged=0.0783 | fraction_flagged=0.0783, jaccard_vs_sklearn=1 | ok |
| dart | istella | algos | 69424 | 23014 | lightgbm-cpu | 35425 | 1.96 | 0.65 | FLIP faster | accuracy=0.9486, logloss=0.134461 | accuracy=0.9487, logloss=0.1341 | accuracy=0.9519, logloss=0.1237 | ok |
| maxabs-scaler | taxi | algos | 13.4 | 7.9 | sklearn-cpu | 12.4 | 1.08 | 0.64 | FLIP faster | output_shape=100000x11 | - | - | ok |
| knn | istella | classical | 1592 | 357 | sklearn-cpu | 566 | 2.82 | 0.63 | FLIP faster | recall_at_k=0.982434, rows_with_repeated_ids=0 | recall_at_k=0.9766, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| knn | taxi | classical | 224 | 257 | sklearn-cpu | 426 | 0.53 | 0.60 |  | recall_at_k=0.999754, rows_with_repeated_ids=0 | recall_at_k=0.9998, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| label-encoder | istella | algos | 297 | 13.4 | sklearn-cpu | 22.5 | 13.21 | 0.60 | FLIP faster | output_shape=100000 | - | - | ok |
| als | taxi-zones | algos | - | 17842 | implicit-cpu | 30318 | - | 0.59 |  | recall_at_10=0.0539953 | - | recall_at_10=0.05679 | ok |
| select-d | taxi-hourly | algos | 7.0 | 2.0 | statsmodels-cpu | 3.4 | 2.06 | 0.59 | FLIP faster | - | - | d_agreement_vs_statsmodels=1 | ok |
| gbdt-ordered | taxi | trees | - | 56105 | catboost-cpu (fill) | 99253 | - | 0.57 |  | logloss=0.52928, auc=0.628294 | - | auc=0.6277 | ok |
| kpss | taxi-hourly | algos | 3.0 | 1.5 | statsmodels-cpu | 2.6 | 1.15 | 0.56 | FLIP faster | stationary_fraction=0.6875 | stationary_fraction=0.6875 | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| gru-clf | taxi-hourly | algos | 1632 | 1650 | torch-eager-fp32 (fill) | 3099 | 0.53 | 0.53 |  | accuracy=0.865668, logloss=0.305841 | accuracy=0.8657, logloss=0.3058 | - | ok |
| gru-reg | synthetic | algos | 1630 | 1629 | torch-eager-fp32 (fill) | 3060 | 0.53 | 0.53 |  | r2=0.981946, rmse=0.155672 | r2=0.9819, rmse=0.1557 | - | ok |
| label-propagation | istella | algos | - | 7714 | sklearn-cpu | 14552 | - | 0.53 |  | accuracy=0.9055 | - | accuracy=0.9055 | ok |
| gru-reg | taxi-hourly | algos | 1643 | 1627 | torch-eager-fp32 (fill) | 3072 | 0.54 | 0.53 |  | r2=0.748219, rmse=0.544182 | r2=0.7482, rmse=0.5442 | - | ok |
| gru-clf | synthetic | algos | 1645 | 1653 | torch-eager-fp32 (fill) | 3137 | 0.52 | 0.53 |  | accuracy=0.971842, logloss=0.0658348 | accuracy=0.9718, logloss=0.06584 | - | ok |
| minmax-scaler | taxi | algos | 22.1 | 9.0 | sklearn-cpu | 17.6 | 1.26 | 0.51 | FLIP faster | - | - | - | ok |
| lr-onecycle | synthetic | algos | 1359 | 55.6 | torch-cpu | 111 | 12.26 | 0.50 | FLIP faster | - | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.951e-08 | ok |
| nearest-centroid | istella | algos | 2896 | 263 | sklearn-cpu | 529 | 5.47 | 0.50 | FLIP faster | accuracy=0.85261, logloss=4.29948 | accuracy=0.8526, logloss=4.299 | accuracy=0.8526, logloss=4.118 | ok |
| enet-cv | taxi | algos | 3173 | 105 | sklearn-cpu | 211 | 15.03 | 0.50 | FLIP faster | r2=0.909004, rmse=4.80449 | r2=0.909, rmse=4.804 | r2=0.909, rmse=4.804 | ok |
| meanshift | istella | algos | 5834 | 214 | sklearn-cpu | 444 | 13.13 | 0.48 | FLIP faster | n_clusters=12, silhouette=0.403452 | ari_vs_ours=1, n_clusters=12, silhouette=0.4034 | ari_vs_ours=1, n_clusters=12, silhouette=0.4034 | ok |
| label-encoder | taxi | algos | 313 | 13.9 | sklearn-cpu | 29.0 | 10.79 | 0.48 | FLIP faster | output_shape=100000 | - | - | ok |
| select-f-regression | taxi | algos | 103 | 15.0 | sklearn-cpu | 31.3 | 3.28 | 0.48 | FLIP faster | n_selected=5 | n_selected=5 | jaccard_vs_sklearn=1, n_selected=5 | ok |
| lda | text | algos | - | 34671 | sklearn-cpu | 72657 | - | 0.48 |  | perplexity=266.719 | - | perplexity=266.9 | ok |
| ard | taxi | algos | 44.3 | 8.3 | sklearn-cpu | 17.5 | 2.53 | 0.47 | FLIP faster | r2=0.90919, rmse=4.79957 | r2=0.9092, rmse=4.8 | r2=0.9092, rmse=4.8 | ok |
| gbdt-rank-yetirank | istella | trees | 25773 | 3350 | lightgbm-cpu (fill) | 7083 | 3.64 | 0.47 | FLIP faster | ndcg10=0.680466, ndcg5=0.614035, map=0.814182 | map=0.8149, ndcg10=0.681, ndcg5=0.6151 | map=0.8584 | ok |
| gbdt-categorical | taxi | trees | 63398 | 27206 | lightgbm-cpu (fill) | 58904 | 1.08 | 0.46 | FLIP faster | logloss=0.528361, auc=0.631219 | auc=0.6302, logloss=0.5286 | auc=0.6327 | ok |
| lasso-cv | taxi | algos | 3170 | 104 | sklearn-cpu | 226 | 14.00 | 0.46 | FLIP faster | r2=0.909038, rmse=4.80359 | r2=0.9091, rmse=4.803 | r2=0.909, rmse=4.804 | ok |
| ridge | taxi | classical2 | 31.5 | 16.2 | sklearn-cpu | 35.7 | 0.88 | 0.45 |  | r2=0.908983, rmse=4.80505 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| bayesian-gmm | istella | algos | - | 52250 | sklearn-cpu (fill) | 116803 | - | 0.45 |  | mean_log_likelihood=175.576 | - | - | ok |
| iterative-imputer | istella | algos | 4605 | 4596 | sklearn-cpu | 10351 | 0.44 | 0.44 |  | masked_rmse=799041 | masked_rmse=7.99e+05 | masked_rmse=8.024e+05 | ok |
| ordinal | taxi | algos | 67.2 | 8.2 | sklearn-cpu | 18.8 | 3.57 | 0.44 | FLIP faster | - | - | - | ok |
| cross-val-score | istella | algos | 3306 | 3075 | sklearn-cpu | 7050 | 0.47 | 0.44 |  | mean_r2=0.333971 | mean_r2=0.3314 | mean_r2=0.3198 | ok |
| isotonic | taxi | algos | 1107 | 38.8 | sklearn-cpu | 90.5 | 12.23 | 0.43 | FLIP faster | r2=0.897069, rmse=5.10987 | r2=0.8971, rmse=5.11 | r2=0.8971, rmse=5.11 | ok |
| select-f-classif | taxi | algos | 128 | 17.5 | sklearn-cpu | 41.1 | 3.12 | 0.43 | FLIP faster | n_selected=5 | n_selected=5 | jaccard_vs_sklearn=1, n_selected=5 | ok |
| select-d | synthetic | algos | 6.6 | 2.2 | statsmodels-cpu | 5.1 | 1.29 | 0.42 | FLIP faster | - | - | d_agreement_vs_statsmodels=1 | ok |
| dynamic-optimized-theta | taxi-hourly | algos | 727 | 562 | statsforecast-cpu | 1352 | 0.54 | 0.42 |  | forecast_rmse=49.0857 | forecast_rmse=49.08 | forecast_rmse=49.31 | ok |
| gbdt-symmetric-1000 | taxi | trees | 19086 | 22353 | catboost-cpu (fill) | 55261 | 0.35 | 0.40 |  | logloss=0.528216, auc=0.631675 | auc=0.6239, logloss=0.5306 | auc=0.6316 | ok |
| gbdt-symmetric | taxi | trees | 9520 | 11253 | catboost-cpu (fill) | 28188 | 0.34 | 0.40 |  | logloss=0.528595, auc=0.630376 | auc=0.6239, logloss=0.5306 | auc=0.6303 | ok |
| dynamic-theta | taxi-hourly | algos | 396 | 303 | statsforecast-cpu | 768 | 0.51 | 0.39 |  | forecast_rmse=49.1014 | forecast_rmse=49.1 | forecast_rmse=49.27 | ok |
| dynamic-optimized-theta | synthetic | algos | 485 | 377 | statsforecast-cpu | 959 | 0.51 | 0.39 |  | forecast_rmse=1.43637 | forecast_rmse=1.436 | forecast_rmse=1.436 | ok |
| ols | taxi | classical | 87.3 | 106 | sklearn-cpu | 274 | 0.32 | 0.39 |  | r2=0.908838, rmse=4.69644 | r2=0.9088, rmse=4.696 | r2=0.7248, rmse=8.159 | ok |
| knn-clf | istella | classical2 | 834 | 109 | sklearn-cpu | 292 | 2.85 | 0.37 | FLIP faster | accuracy=0.92625 | accuracy=0.9263 | accuracy=0.9263 | ok |
| lasso-lars | taxi | algos | 184 | 12.8 | sklearn-cpu | 34.6 | 5.32 | 0.37 | FLIP faster | r2=0.908998, rmse=4.80466 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.804 | ok |
| auto-theta | synthetic | algos | 962 | 718 | statsforecast-cpu | 1953 | 0.49 | 0.37 |  | forecast_rmse=1.43688 | forecast_rmse=1.438 | forecast_rmse=1.438 | ok |
| knn-reg | istella | classical2 | 823 | 105 | sklearn-cpu | 285 | 2.88 | 0.37 | FLIP faster | r2=0.418145, rmse=0.625388 | r2=0.4181, rmse=0.6254 | r2=0.4181, rmse=0.6254 | ok |
| gbdt-rank-pairlogit | istella | trees | 3997 | 3064 | xgboost-cpu (fill) | 8355 | 0.48 | 0.37 |  | ndcg10=0.719953, ndcg5=0.6504, map=0.854545 | map=0.8414, ndcg10=0.7093, ndcg5=0.6398 | map=0.8728 | ok |
| ocsvm | taxi | algos | 246 | 65.0 | sklearn-cpu | 181 | 1.36 | 0.36 | FLIP faster | fraction_flagged=0.1361 | fraction_flagged=0.1361 | fraction_flagged=0.1361, jaccard_vs_sklearn=1 | ok |
| ivf-pq | istella | algos | 6215 | 2503 | faiss-cpu | 7008 | 0.89 | 0.36 |  | recall_at_10=0.7561 | recall_at_10=0.5995 | recall_at_10=0.803 | ok |
| lars | taxi | algos | 185 | 12.1 | sklearn-cpu | 34.2 | 5.40 | 0.35 | FLIP faster | r2=0.908983, rmse=4.80505 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| ivf-filter | istella | algos | 6190 | 2516 | faiss-cpu | 7234 | 0.86 | 0.35 |  | recall_at_10=0.801 | recall_at_10=0.6527 | recall_at_10=0.8419 | ok |
| ivf-refine | istella | algos | 6078 | 2507 | faiss-cpu | 7322 | 0.83 | 0.34 |  | recall_at_10=0.982 | recall_at_10=0.8622 | recall_at_10=0.9934 | ok |
| elasticnet | taxi | classical2 | 61.1 | 13.7 | sklearn-cpu | 40.6 | 1.50 | 0.34 | FLIP faster | r2=0.907378, rmse=4.84724 | r2=0.9074, rmse=4.847 | r2=0.9074, rmse=4.847 | ok |
| gbdt-lossguide | istella | trees | 106296 | 18364 | lightgbm-cpu (fill) | 55012 | 1.93 | 0.33 | FLIP faster | logloss=0.148902, auc=0.98367 | auc=0.9838, logloss=0.1488 | auc=0.9838 | ok |
| gbdt-multiclass | taxi | trees | 14685 | 14938 | xgboost-cpu (fill) | 45500 | 0.32 | 0.33 |  | mlogloss=1.01259, accuracy=0.59938 | accuracy=0.5967, mlogloss=1.023 | accuracy=0.6011 | ok |
| sparse-pca | istella | algos | - | 3197 | sklearn-cpu | 9904 | - | 0.32 |  | component_sparsity=0.305682, relative_reconstruction_error=0.75021 | - | component_sparsity=0.3057, relative_reconstruction_error=0.7502 | ok |
| select-r-regression | taxi | algos | 104 | 10.1 | sklearn-cpu | 31.8 | 3.25 | 0.32 | FLIP faster | n_selected=5 | n_selected=5 | jaccard_vs_sklearn=1, n_selected=5 | ok |
| optimized-theta | taxi-hourly | algos | 576 | 398 | statsforecast-cpu | 1290 | 0.45 | 0.31 |  | forecast_rmse=49.1549 | forecast_rmse=49.15 | forecast_rmse=49.36 | ok |
| ridge-clf | istella | algos | 9790 | 2153 | sklearn-cpu | 7044 | 1.39 | 0.31 | FLIP faster | accuracy=0.91054 | accuracy=0.8943 | accuracy=0.9105 | ok |
| optimized-theta | synthetic | algos | 311 | 227 | statsforecast-cpu | 748 | 0.42 | 0.30 |  | forecast_rmse=1.43851 | forecast_rmse=1.44 | forecast_rmse=1.438 | ok |
| minmax-scaler | istella | algos | 359 | 19.1 | sklearn-cpu | 63.0 | 5.70 | 0.30 | FLIP faster | - | - | - | ok |
| onehot | taxi | algos | 71.8 | 5.6 | sklearn-cpu | 19.0 | 3.78 | 0.29 | FLIP faster | - | - | - | ok |
| gbdt-symmetric | istella | trees | 11943 | 16955 | catboost-cpu (fill) | 60136 | 0.20 | 0.28 |  | logloss=0.186714, auc=0.980109 | auc=0.9756, logloss=0.2112 | auc=0.9799 | ok |
| lstsq | istella | algos | 5886 | 844 | numpy-cpu | 3086 | 1.91 | 0.27 | FLIP faster | relative_residual=0.849956 | relative_residual=0.85 | relative_residual=0.8733 | ok |
| gbdt-symmetric-1000 | istella | trees | 21364 | 32889 | catboost-cpu (fill) | 120805 | 0.18 | 0.27 |  | logloss=0.1704, auc=0.982434 | auc=0.9756, logloss=0.2112 | auc=0.9823 | ok |
| lasso | taxi | classical2 | 63.8 | 12.8 | sklearn-cpu | 47.9 | 1.33 | 0.27 | FLIP faster | r2=0.908995, rmse=4.80474 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| ols | istella | classical | 1332 | 850 | sklearn-cpu | 3245 | 0.41 | 0.26 |  | r2=0.331943, rmse=0.682027 | r2=0.3211, rmse=0.6875 | r2=0.001881, rmse=0.8337 | ok |
| dynamic-theta | synthetic | algos | 116 | 85.9 | statsforecast-cpu | 333 | 0.35 | 0.26 |  | forecast_rmse=1.43726 | forecast_rmse=1.437 | forecast_rmse=1.437 | ok |
| gpr | istella | classical2 | 218 | 130 | sklearn-cpu | 523 | 0.42 | 0.25 |  | mean_log_predictive_density=-9.28532, r2=0.235374, rmse=0.760426 | mean_log_predictive_density=-9.285, r2=0.2354, rmse=0.7604 | mean_log_predictive_density=-9.287, r2=0.2354, rmse=0.7604 | ok |
| gbdt-lossguide | taxi | trees | 44556 | 12721 | lightgbm-cpu (fill) | 51864 | 0.86 | 0.25 |  | logloss=0.528132, auc=0.631865 | auc=0.631, logloss=0.5283 | auc=0.6322 | ok |
| gbdt-multiclass | istella | trees | 21754 | 25728 | xgboost-cpu (fill) | 105535 | 0.21 | 0.24 |  | mlogloss=0.258413, accuracy=0.907556 | accuracy=0.9033, mlogloss=0.2809 | accuracy=0.9101 | ok |
| ridge-clf | taxi | algos | 410 | 19.0 | sklearn-cpu | 77.9 | 5.27 | 0.24 | FLIP faster | accuracy=0.76357 | accuracy=0.7636 | accuracy=0.7636 | ok |
| lda-clf | taxi | algos | 87.4 | 43.1 | sklearn-cpu | 186 | 0.47 | 0.23 |  | accuracy=0.76253, logloss=0.539763 | accuracy=0.7626, logloss=0.5397 | accuracy=0.7625, logloss=0.5398 | ok |
| linearsvc | taxi | classical2 | 112 | 104 | sklearn-cpu | 451 | 0.25 | 0.23 |  | accuracy=0.76333 | accuracy=0.7633 | accuracy=0.7636 | ok |
| categorical-nb | taxi | algos | 139 | 18.0 | sklearn-cpu | 78.3 | 1.77 | 0.23 | FLIP faster | accuracy=0.76585, logloss=0.538866 | accuracy=0.7659, logloss=0.5389 | accuracy=0.7659, logloss=0.5389 | ok |
| ovr | taxi | algos | 458 | 324 | sklearn-cpu | 1461 | 0.31 | 0.22 |  | accuracy=0.47894 | accuracy=0.4789 | accuracy=0.4789 | ok |
| arima | synthetic | classical2 | 108 | 80.5 | statsmodels-cpu | 364 | 0.30 | 0.22 |  | forecast_rmse=1.51554, insample_rmse=0.999341, mean_aic=5680.97, mean_llf=-2836.49 | forecast_rmse=1.516, insample_rmse=0.9993, mean_aic=5681, mean_llf=-2836 | forecast_rmse=1.515, insample_rmse=0.9993, mean_aic=5681, mean_llf=-2836 | ok |
| nearest-centroid | taxi | algos | 471 | 20.9 | sklearn-cpu | 96.9 | 4.86 | 0.22 | FLIP faster | accuracy=0.6667, logloss=0.781905 | accuracy=0.6667, logloss=0.7822 | accuracy=0.6667, logloss=0.7817 | ok |
| bayesian-ridge | taxi | algos | 440 | 15.3 | sklearn-cpu | 73.2 | 6.01 | 0.21 | FLIP faster | r2=0.908983, rmse=4.80505 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| iterative-imputer | taxi | algos | 219 | 218 | sklearn-cpu | 1049 | 0.21 | 0.21 |  | masked_rmse=4.69397 | masked_rmse=4.694 | masked_rmse=4.694 | ok |
| dart | taxi | algos | 10520 | 5378 | lightgbm-cpu | 26498 | 0.40 | 0.20 |  | accuracy=0.76815, logloss=0.529149 | accuracy=0.7683, logloss=0.5291 | accuracy=0.7682, logloss=0.5291 | ok |
| label-propagation | taxi | algos | 6002 | 2090 | sklearn-cpu | 10366 | 0.58 | 0.20 |  | accuracy=0.7016 | accuracy=0.7016 | accuracy=0.7016 | ok |
| rf | taxi | trees | 10796 | 10708 | lightgbm-cpu | 53864 | 0.20 | 0.20 |  | logloss=0.525953, auc=0.617838 | auc=0.6178, logloss=0.526 | auc=0.617, logloss=0.5264 | ok |
| garch | taxi-hourly | algos | 25.4 | 23.6 | arch-cpu | 125 | 0.20 | 0.19 |  | mean_llf=-1131.91 | mean_llf=-1133 | mean_llf=-1130 | ok |
| dart-reg | taxi | algos | 9932 | 5199 | lightgbm-cpu | 27650 | 0.36 | 0.19 |  | r2=0.925497, rmse=4.34734 | r2=0.9255, rmse=4.347 | r2=0.9263, rmse=4.325 | ok |
| knn-clf | taxi | classical2 | 48.5 | 28.9 | sklearn-cpu | 154 | 0.32 | 0.19 |  | accuracy=0.74175 | accuracy=0.7418 | accuracy=0.7418 | ok |
| categorical-nb | istella | algos | 144 | 15.8 | sklearn-cpu | 84.4 | 1.71 | 0.19 | FLIP faster | accuracy=0.83885, logloss=0.412625 | accuracy=0.8388, logloss=0.4126 | accuracy=0.8388, logloss=0.4126 | ok |
| poisson | taxi | algos | - | 79.7 | sklearn-cpu (fill) | 442 | - | 0.18 |  | r2=0.035748, rmse=15.6398 | - | - | ok |
| factor-analysis | istella | algos | - | 6172 | sklearn-cpu | 35135 | - | 0.18 |  | mean_log_likelihood=89.0315 | - | mean_log_likelihood=98.12 | ok |
| croston-optimized | taxi-hourly | algos | 10.4 | 9.8 | statsforecast-cpu | 60.5 | 0.17 | 0.16 |  | forecast_rmse=1.39825 | forecast_rmse=1.398 | forecast_rmse=1.398 | ok |
| garch | synthetic | algos | 23.4 | 16.5 | arch-cpu | 106 | 0.22 | 0.16 |  | mean_llf=-1938.22 | mean_llf=-1938 | mean_llf=-1938 | ok |
| adaboost-reg | taxi | algos | 2609 | 1887 | sklearn-cpu | 12197 | 0.21 | 0.15 |  | r2=-0.605962, rmse=20.1839 | r2=0.2164, rmse=14.1 | r2=0.5639, rmse=10.52 | ok |
| rf | istella | trees | 13810 | 10162 | lightgbm-cpu | 67210 | 0.21 | 0.15 |  | logloss=0.182017, auc=0.945385 | auc=0.9454, logloss=0.182 | auc=0.9454, logloss=0.1954 | ok |
| croston-optimized | synthetic | algos | 5.9 | 9.4 | statsforecast-cpu | 62.3 | 0.09 | 0.15 |  | forecast_rmse=1.67554 | forecast_rmse=1.675 | forecast_rmse=1.675 | ok |
| iforest | taxi | trees | 119 | 85.8 | sklearn-iforest-cpu (fill) | 575 | 0.21 | 0.15 |  | auc=0.551846 | auc=0.5518 | auc=0.5528 | ok |
| tweedie | taxi | algos | - | 48.9 | sklearn-cpu (fill) | 333 | - | 0.15 |  | r2=-10.0744, rmse=53.0026 | - | - | ok |
| lda-clf | istella | algos | 19228 | 533 | sklearn-cpu | 3739 | 5.14 | 0.14 | FLIP faster | accuracy=0.91314, logloss=0.235662 | accuracy=0.909, logloss=0.2644 | accuracy=0.9011, logloss=0.4492 | ok |
| stacking-reg | taxi | algos | - | 853 | sklearn-cpu | 6008 | - | 0.14 |  | r2=0.919717, rmse=4.51282 | - | r2=0.9325, rmse=4.137 | ok |
| lr-step | synthetic | algos | 11.7 | 11.9 | torch-cpu | 85.1 | 0.14 | 0.14 |  | - | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=1.49e-08 | ok |
| et | taxi | trees | 3376 | 2854 | sklearn-et-cpu | 20629 | 0.16 | 0.14 |  | logloss=0.526142, auc=0.618907 | auc=0.6189, logloss=0.5261 | auc=0.619, logloss=0.526 | ok |
| gamma | taxi | algos | - | 47.7 | sklearn-cpu (fill) | 360 | - | 0.13 |  | r2=-231.912, rmse=243.071 | - | - | ok |
| et | istella | trees | 4360 | 4034 | sklearn-et-cpu | 30617 | 0.14 | 0.13 |  | logloss=0.189989, auc=0.937987 | auc=0.938, logloss=0.19 | auc=0.9379, logloss=0.1901 | ok |
| logreg-cv | taxi | algos | - | 230 | sklearn-cpu | 1758 | - | 0.13 |  | accuracy=0.76332, logloss=0.538985 | - | accuracy=0.7633, logloss=0.539 | ok |
| multioutput-clf | taxi | algos | 228 | 87.7 | sklearn-cpu | 686 | 0.33 | 0.13 |  | accuracy=0.86356 | accuracy=0.8636 | accuracy=0.8636 | ok |
| logreg | istella | classical2 | 3906 | 3609 | sklearn-cpu | 28852 | 0.14 | 0.13 |  | accuracy=0.92456, logloss=0.181239, nonfinite_proba_rows=0 | accuracy=0.9246, logloss=0.1812, nonfinite_proba_rows=0 | accuracy=0.9245, logloss=0.1813, nonfinite_proba_rows=0 | ok |
| knn-reg | taxi | classical2 | 22.7 | 16.1 | sklearn-cpu | 131 | 0.17 | 0.12 |  | r2=0.937323, rmse=3.84203 | r2=0.9373, rmse=3.842 | r2=0.9373, rmse=3.842 | ok |
| cross-val-score | taxi | algos | 184 | 154 | sklearn-cpu | 1282 | 0.14 | 0.12 |  | mean_r2=0.937955 | mean_r2=0.938 | mean_r2=0.938 | ok |
| theta | taxi-hourly | algos | 1747 | 20.3 | statsmodels-cpu | 171 | 10.20 | 0.12 | FLIP faster | forecast_rmse=49.0206 | forecast_rmse=49.28 | forecast_rmse=49.31 | ok |
| huber | taxi | algos | 168025 | 215 | sklearn-cpu | 1855 | 90.57 | 0.12 | FLIP faster | r2=0.900215, rmse=5.03117 | r2=0.9002, rmse=5.031 | r2=0.9002, rmse=5.031 | ok |
| ridge-cv | istella | algos | - | 14955 | sklearn-cpu | 134463 | - | 0.11 |  | r2=0.328684, rmse=0.684422 | - | r2=0.3287, rmse=0.6844 | ok |
| multioutput-reg | istella | algos | 1823 | 1419 | sklearn-cpu | 13402 | 0.14 | 0.11 |  | r2=0.455327 | r2=0.4399 | r2=0.4553 | ok |
| logreg | taxi | classical2 | 119 | 32.7 | sklearn-cpu | 310 | 0.38 | 0.11 |  | accuracy=0.76335, logloss=0.538985, nonfinite_proba_rows=0 | accuracy=0.7633, logloss=0.539, nonfinite_proba_rows=0 | accuracy=0.7633, logloss=0.539, nonfinite_proba_rows=0 | ok |
| lasso-cv | istella | algos | 70206 | 556 | sklearn-cpu | 5483 | 12.80 | 0.10 | FLIP faster | r2=0.325504, rmse=0.686041 | r2=0.3103, rmse=0.6937 | r2=0.3108, rmse=0.6935 | ok |
| prophet | synthetic | algos | 70.6 | 44.8 | prophet-cpu | 452 | 0.16 | 0.10 |  | forecast_rmse=1.01491 | forecast_rmse=1.015 | forecast_rmse=1.015 | ok |
| stacking-clf | taxi | algos | 1080 | 702 | sklearn-cpu | 7095 | 0.15 | 0.10 |  | accuracy=0.76792, logloss=0.536364 | accuracy=0.7679, logloss=0.5364 | accuracy=0.7553, logloss=0.5478 | ok |
| damped-ets | synthetic | algos | 530 | 16.1 | statsforecast-cpu | 167 | 3.17 | 0.10 | FLIP faster | forecast_rmse=13.9307 | forecast_rmse=13.93 | forecast_rmse=13.95 | ok |
| damped-ets | taxi-hourly | algos | 612 | 17.3 | statsforecast-cpu | 185 | 3.31 | 0.09 | FLIP faster | forecast_rmse=96.6846 | forecast_rmse=96.69 | forecast_rmse=96.69 | ok |
| multioutput-clf | istella | algos | 2212 | 2087 | sklearn-cpu | 22340 | 0.10 | 0.09 |  | accuracy=0.959175 | accuracy=0.9592 | accuracy=0.9592 | ok |
| calibrated | taxi | algos | 1330 | 48.8 | sklearn-cpu | 556 | 2.39 | 0.09 | FLIP faster | accuracy=0.75533, logloss=0.550718 | accuracy=0.7553, logloss=0.5507 | accuracy=0.7553, logloss=0.5507 | ok |
| logreg-cv | istella | algos | - | 6186 | sklearn-cpu | 72840 | - | 0.08 |  | accuracy=0.92454, logloss=0.181358 | - | accuracy=0.9246, logloss=0.1814 | ok |
| enet-cv | istella | algos | 74148 | 581 | sklearn-cpu | 6979 | 10.62 | 0.08 | FLIP faster | r2=0.326794, rmse=0.685384 | r2=0.3166, rmse=0.6906 | r2=0.3173, rmse=0.6902 | ok |
| ard | istella | algos | 47943 | 863 | sklearn-cpu | 10421 | 4.60 | 0.08 | FLIP faster | r2=-0.138726, rmse=0.891395 | r2=-0.1235, rmse=0.8854 | r2=0.3274, rmse=0.6851 | ok |
| elasticnet | istella | classical2 | 392 | 150 | sklearn-cpu | 1809 | 0.22 | 0.08 |  | r2=0.260922, rmse=0.718134 | r2=0.2616, rmse=0.7178 | r2=0.2609, rmse=0.7181 | ok |
| ridge | istella | classical2 | 879 | 538 | sklearn-cpu | 6717 | 0.13 | 0.08 |  | r2=0.328693, rmse=0.684417 | r2=0.3205, rmse=0.6886 | r2=0.3287, rmse=0.6844 | ok |
| prophet | taxi-hourly | algos | 72.7 | 43.2 | prophet-cpu | 549 | 0.13 | 0.08 |  | forecast_rmse=32.0407 | forecast_rmse=32.02 | forecast_rmse=32.03 | ok |
| bayesian-ridge | istella | algos | 92368 | 494 | sklearn-cpu | 6831 | 13.52 | 0.07 | FLIP faster | r2=0.328404, rmse=0.684565 | r2=-4.169e+04, rmse=170.6 | r2=-890.2, rmse=24.94 | ok |
| lda | taxi-zones | algos | 41177 | 1509 | sklearn-cpu | 20926 | 1.97 | 0.07 | FLIP faster | perplexity=45.2218 | perplexity=45.22 | perplexity=44.9 | ok |
| calibrated | istella | algos | 2291 | 193 | sklearn-cpu | 2784 | 0.82 | 0.07 |  | accuracy=0.88509, logloss=0.289824 | accuracy=0.8851, logloss=0.2898 | accuracy=0.8851, logloss=0.2898 | ok |
| stl | synthetic | algos | 359 | 7.3 | statsmodels-cpu | 105 | 3.43 | 0.07 | FLIP faster | residual_std=0.783175 | residual_std=0.7832 | residual_std=0.7832 | ok |
| ovr | istella | algos | 5375 | 5226 | sklearn-cpu | 75802 | 0.07 | 0.07 |  | accuracy=0.8927 | accuracy=0.8927 | accuracy=0.8927 | ok |
| stacking-reg | istella | algos | - | 10306 | sklearn-cpu | 153421 | - | 0.07 |  | r2=0.448252, rmse=0.620485 | - | r2=0.4476, rmse=0.6208 | ok |
| qda | istella | algos | 16290 | 429 | sklearn-cpu | 6397 | 2.55 | 0.07 | FLIP faster | accuracy=0.88049, logloss=3.60871 | accuracy=0.8661, logloss=4.053 | accuracy=0.8805, logloss=3.477 | ok |
| croston-sba | synthetic | algos | 2.9 | 2.2 | statsforecast-cpu | 34.5 | 0.08 | 0.06 |  | forecast_rmse=1.67446 | forecast_rmse=1.675 | forecast_rmse=1.675 | ok |
| stacking-clf | istella | algos | 4053 | 9413 | sklearn-cpu | 149799 | 0.03 | 0.06 |  | accuracy=0.92997, logloss=0.193693 | accuracy=0.93, logloss=0.1937 | accuracy=0.93, logloss=0.1939 | ok |
| stl | taxi-hourly | algos | 359 | 7.0 | statsmodels-cpu | 119 | 3.02 | 0.06 | FLIP faster | residual_std=18.3125 | residual_std=18.31 | residual_std=18.31 | ok |
| adaboost-clf | taxi | algos | 3118 | 1616 | sklearn-cpu | 27304 | 0.11 | 0.06 |  | accuracy=0.76523, logloss=0.543302 | accuracy=0.7652, logloss=0.5432 | accuracy=0.7654, logloss=0.5406 | ok |
| croston-sba | taxi-hourly | algos | 2.5 | 2.3 | statsforecast-cpu | 47.9 | 0.05 | 0.05 |  | forecast_rmse=1.38668 | forecast_rmse=1.387 | forecast_rmse=1.387 | ok |
| affinity-prop | istella | algos | 964 | 265 | sklearn-cpu | 5734 | 0.17 | 0.05 |  | n_clusters=342, silhouette=0.0897635 | ari_vs_ours=1, n_clusters=342, silhouette=0.08976 | ari_vs_ours=1, n_clusters=342, silhouette=0.08976 | ok |
| voting-reg | taxi | algos | - | 56.2 | sklearn-cpu | 1272 | - | 0.04 |  | r2=0.919181, rmse=4.52786 | - | r2=0.9246, rmse=4.372 | ok |
| croston | taxi-hourly | algos | 3.1 | 2.3 | statsforecast-cpu | 54.2 | 0.06 | 0.04 |  | forecast_rmse=1.39026 | forecast_rmse=1.39 | forecast_rmse=1.39 | ok |
| lasso | istella | classical2 | 395 | 143 | sklearn-cpu | 3839 | 0.10 | 0.04 |  | r2=0.310839, rmse=0.693459 | r2=0.3105, rmse=0.6936 | r2=0.3108, rmse=0.6935 | ok |
| ridge-cv | taxi | algos | - | 33.1 | sklearn-cpu | 946 | - | 0.04 |  | r2=0.908983, rmse=4.80505 | - | r2=0.909, rmse=4.805 | ok |
| bootstrap | istella | algos | 10.2 | 14.2 | scipy-cpu | 525 | 0.02 | 0.03 |  | standard_error=0.00598712, ci_low=0.27125, ci_high=0.29485 | ci_high=0.2949, ci_low=0.2712, standard_error=0.005987 | ci_high=0.2955, ci_low=0.2717, standard_error=0.006044 | ok |
| voting-reg | istella | algos | - | 932 | sklearn-cpu | 37765 | - | 0.02 |  | r2=0.406572, rmse=0.643494 | - | r2=0.4067, rmse=0.6434 | ok |
| croston | synthetic | algos | 2.9 | 1.9 | statsforecast-cpu | 91.7 | 0.03 | 0.02 |  | forecast_rmse=1.67484 | forecast_rmse=1.675 | forecast_rmse=1.675 | ok |
| poisson | istella | algos | - | 2168 | sklearn-cpu (fill) | 110378 | - | 0.02 |  | r2=0.243601, rmse=0.7265 | - | - | ok |
| bootstrap | taxi | algos | 13.3 | 9.4 | scipy-cpu | 525 | 0.03 | 0.02 |  | standard_error=0.117837, ci_low=18.2494, ci_high=18.7174 | ci_high=18.72, ci_low=18.25, standard_error=0.1178 | ci_high=18.71, ci_low=18.25, standard_error=0.1176 | ok |
| permutation-test | istella | algos | - | 53.5 | scipy-cpu | 3042 | - | 0.02 |  | pvalue=0.1802, statistic=0.0112 | - | pvalue=0.1844, statistic=0.0112 | ok |
| permutation-test | taxi | algos | - | 52.9 | scipy-cpu | 3055 | - | 0.02 |  | pvalue=0.0006, statistic=-0.576702 | - | pvalue=0.001, statistic=-0.5767 | ok |
| huber | istella | algos | 48741 | 738 | sklearn-cpu | 45995 | 1.06 | 0.02 | FLIP faster | r2=-0.00692323, rmse=0.838221 | r2=-0.008227, rmse=0.8388 | r2=-0.01018, rmse=0.8396 | ok |
| gamma | istella | algos | - | 1551 | sklearn-cpu (fill) | 107798 | - | 0.01 |  | r2=0.218164, rmse=0.738615 | - | - | ok |
| tweedie | istella | algos | - | 1547 | sklearn-cpu (fill) | 108690 | - | 0.01 |  | r2=-24.4124, rmse=4.21098 | - | - | ok |
| adaboost-reg | istella | algos | 6071 | 1820 | sklearn-cpu | 183985 | 0.03 | 0.01 |  | r2=0.242748, rmse=0.72691 | r2=0.2389, rmse=0.7288 | r2=0.1676, rmse=0.7621 | ok |
| optics | istella | algos | 514 | 261 | sklearn-cpu | 30143 | 0.02 | 0.01 |  | n_clusters=20, silhouette=-0.287356 | ari_vs_ours=1, n_clusters=20, silhouette=-0.2874 | ari_vs_ours=0.9846, n_clusters=20, silhouette=-0.2858 | ok |
| meanshift | taxi | algos | 538 | 43.6 | sklearn-cpu | 9370 | 0.06 | 0.00 |  | n_clusters=122, silhouette=0.246631 | ari_vs_ours=1, n_clusters=122, silhouette=0.2466 | ari_vs_ours=1, n_clusters=122, silhouette=0.2466 | ok |
| kde | istella | classical | 1539 | 142 | sklearn-cpu | 54651 | 0.03 | 0.00 |  | mean_log_likelihood=-222.271, rows_without_density=0 | mean_log_likelihood=-222.3, rows_without_density=0 | mean_log_likelihood=-227, rows_without_density=0 | ok |
| linearsvr | taxi | classical2 | 183 | 113 | sklearn-cpu | 83935 | 0.00 | 0.00 |  | r2=0.899807, rmse=5.04144 | r2=0.8998, rmse=5.041 | r2=0.8998, rmse=5.042 | ok |
| kde | taxi | classical | 60.3 | 8.8 | sklearn-cpu | 6992 | 0.01 | 0.00 |  | mean_log_likelihood=-14.8264, rows_without_density=0 | mean_log_likelihood=-14.83, rows_without_density=0 | mean_log_likelihood=-14.83, rows_without_density=0 | ok |
| adaboost-clf | istella | algos | 15452 | 4763 | - | - | - | - |  | accuracy=0.93715, logloss=0.431913 | accuracy=0.935, logloss=0.4387 | - | ok |
| als | text | algos | - | 127109 | - | - | - | - |  | recall_at_10=0.548738 | - | - | ok |
| gbdt-categorical | taxicat | trees | - | 37105 | - | - | - | - |  | logloss=0.528399, auc=0.630905 | - | - | ok |
| gbdt-ordered | istella | trees | - | 75359 | - | - | - | - |  | logloss=0.190603, auc=0.979518 | - | - | ok |
| gmm | taxi | classical2 | - | - | - | - | - | - |  | - | - | - | error: GaussianMixture ill-defined empirical covariance (no prior FAST time; sklearn refuses too) |
| knn-imputer | istella | algos | - | 32.6 | - | - | - | - |  | masked_rmse=323953 | - | - | ok |
| lamb | synthetic | algos | - | 213 | - | - | - | - |  | - | - | - | ok |
| lars | istella | algos | 5583 | 845 | - | - | - | - |  | r2=0.328662, rmse=0.684433 | r2=0.2157, rmse=0.7398 | - | ok |
| linearsvc | istella | classical2 | 772 | 740 | - | - | - | - |  | accuracy=0.92312 | accuracy=0.9232 | - | ok |
| linearsvr | istella | classical2 | 940 | 217 | - | - | - | - |  | r2=-0.106761, rmse=0.878794 | r2=-0.1067, rmse=0.8788 | - | ok |
| lion | synthetic | algos | 137 | 201 | - | - | - | - |  | - | - | - | ok |
| quantile | istella | algos | - | 764 | - | - | - | - |  | r2=-0.0399981, rmse=0.851877 | - | - | ok |

Sources: before = M3 0.8.34 board (classical), M3 2026-09-29 board FAST cells (trees); job tags afb-eighscope, afb10-ols, afb11-arima-c2, afb12-seq-algos-1, afb12-seq-algos-2, afb12-seq-algos-3, afb12-seq-algos-4, afb13-forest, afb14-lossguide, afb15-prophet, afb16-garch-croston, afb17-gram-algos, afb18-cluster, afb19-prep, afb2-algos-2, afb2-classical-4, afb2-trees-1b, afb20-eigh, afb21-garch, afb23-opt, afb24-knn, afb25-kde, afb26-moe, afb27-ridge, afb28-linear, afb29-lp, afb3-cat-taxicat, afb3-trees-fast, afb30-cat, afb31-yeti, afb4-grp-ocsvm-main, afb5-merged-algos, afb5-merged-trees, afb5b-dart, afb6-te-algos, afb9-dw2, ann-ivffilter-devcb-istella-b, ann-ivfpq-devcb-istella-b, ann-ivfrefine-devcb-istella-b, bayesq-istella, cluster2-ap-split-istella-b, cluster2-optics-devorder-istella-b, gap-dwtaxi, gaparima-async-synthetic, gaparima-async-taxi, gapcagra-ivfgxsi-istella, gapcagra-seedsiters-taxi, gapcls1-ardall-taxi, gapcls1-brall-taxi, gapcls1-k64chk-istella, gapcls1-nclabels-taxi, gapcls1-rccodes-taxi, gapcls2-all3-ocsvm-taxi, gapcls2-devscan-grp-istella, gapcls2-devscan-grp-taxi, gapcls2-fusedpool-minmax-istella, gapcls2-pool-mbk-istella, gapcls2-pool-mbk-taxi, gapcls2-present-onehot-taxi, gapcls2-present-ordinal-taxi, gaptsa-kpsspack-kpss-synthetic, gaptsa-kpsspack-kpss-taxi-hourly, gaptsa-seldfused-selectd-synthetic, gaptsa-seldfused-selectd-taxi-hourly, gaptsa-thetaspec-theta-taxi-hourly, gaptsa-varonecopy-var-synthetic, gaptsa-varonecopy-var-taxi-hourly, nb-lda-fused-zones, nofast-adam-synthetic, nofast-adamw-synthetic, nofast-als-taxi-zones, nofast-als-text, nofast-bayesian-gmm-istella, nofast-bootstrap-istella, nofast-bootstrap-taxi, nofast-cross-val-score-istella-b, nofast-cross-val-score-taxi-b, nofast-factor-analysis-istella-b, nofast-gamma-istella-b, nofast-gamma-taxi-b, nofast-knn-imputer-istella-b, nofast-label-spreading-istella-b, nofast-lof-istella-b, nofast-logreg-cv-istella-b, nofast-logreg-cv-taxi-b, nofast-permutation-test-istella-b, nofast-permutation-test-taxi-b, nofast-poisson-istella-b, nofast-poisson-taxi-b, nofast-quantile-istella-b, nofast-resample-istella-b, nofast-resample-taxi-b, nofast-sparse-pca-istella-b, nofast-svgp-taxi-b, nofast-tweedie-istella-b, nofast-tweedie-taxi-b, nofast-voting-reg-istella-b, nofast-voting-reg-taxi-b, regress-adafactor-pipe, regress-dotm-snap, regress-layernorm-pipe, retime-additive-chi2-istella-b, retime-additive-chi2-taxi-b, retime-bisecting-kmeans-istella-b, retime-bisecting-kmeans-taxi-b, retime-categorical-nb-istella-b, retime-categorical-nb-taxi-b, retime-cholesky-synthetic-b, retime-complement-nb-text-b, retime-connected-components-istella-b, retime-connected-components-taxi-b, retime-elliptic-envelope-taxi-b, retime-huber-istella-b, retime-huber-taxi-b, retime-incremental-pca-taxi-b, retime-isotonic-istella-b, retime-isotonic-taxi-b, retime-kernel-pca-istella-b, retime-kernel-pca-taxi-b, retime-kernel-shap-istella-b, retime-knn-imputer-taxi-b, retime-label-binarizer-istella-b, retime-label-binarizer-taxi-b, retime-label-encoder-istella-b, retime-label-encoder-taxi-b, retime-lle-istella-b, retime-lle-taxi-b, retime-lstsq-istella-b, retime-lu-factor-synthetic-b, retime-lu-solve-synthetic-b, retime-maxabs-scaler-istella-b, retime-maxabs-scaler-taxi-b, retime-mb-dict-learning-istella-b, sel-fcls-taxi, sel-freg-taxi-b, sel-rreg-taxi, stl-main-synthetic, tsa2-stl-taxi-hourly.

## Quality flags (M3 manager, 2026-10-02, updated 2026-10-03 refresh 9 + A/B updates)

- **bayesian-ridge istella:** NaN fixed by the guard; FAST quality fixed in main 9525ed958 (A/B bayesq-istella): r2 0.3284 vs sklearn float64 0.3287 (sklearn-cpu float32 board cell r2 -890).
- **ard istella: r2 -0.139 vs sklearn 0.327.** FAST was already below sklearn before this pass (-0.124); the row-pass arm
  gives -0.122. Speed is real, quality is not at the opponent's level: open.
- gbdt-rank-yetirank istella re-timed in refresh 9 (job afb31-yeti, head eb3fca1ac, YetiRank sort): 3350 ms, was refused;
  ratio 0.47 vs lightgbm-cpu (fill) 7083 ms; ndcg10 0.6805 vs 0.681 before.
- gbdt-categorical taxi re-timed in refresh 9 (job afb30-cat, head eb3fca1ac, categorical CTR freq): 27206 ms, was 32925 ms;
  ratio 0.46 vs lightgbm-cpu (fill) 58904 ms; auc 0.6312 (was 0.6310).
- Slower than before within this refresh: knn istella 1592 -> 1805 ms, ocsvm taxi 246 -> 372 ms. Before cells come
  from older boards (Sept 29 trees, 0.8.34 classical); not yet re-checked.
- Forest scans (main 69f7a41fd, job afb13-forest): rf taxi 11364 -> 10708 ms, rf istella 14232 -> 10162 ms,
  et taxi 3020 -> 2854 ms, et istella 4305 -> 4034 ms; rf and et now beat their Sept 29 FAST cells. Quality unchanged.
- Lossguide exact batch width 128 by default (main 9722e5a2b, job afb14-lossguide): gbdt-lossguide taxi 13102 -> 12721 ms,
  istella 18525 -> 18364 ms. Quality within 0.1% (logloss 0.528132 / 0.148902).
- OLS: FAST on Apple uses equilibrated normal equations since 94cb5ba59 (IDENTICAL keeps TSQR; LinearRegression ID check owed).
- Opponents that refused in the M3 fill (no time, not on this table): gmm taxi (sklearn: ill-defined covariance),
  lars istella (sklearn: OverflowError).
- Prophet cooperative team fit default since 8c2b7de51. Job afb15-prophet: 44.8 / 43.2 ms (synthetic / taxi-hourly),
  was 429 / 358 ms; forecast_rmse 1.0149 / 32.041 vs prophet-cpu 1.015 / 32.03; ratio 0.10 / 0.08 vs prophet-cpu.
- Optimizers re-timed in refresh 8 (job afb23-opt, head 5990c5946): adamax, adagrad, rmsprop, nadam, lion, lamb
  FAST 195-213 ms (was 318-339 ms in refresh 7), still slower than 0.8.34 FAST (129-178 ms) and behind torch-eager-fp32
  (14-33 ms on adagrad, rmsprop, adamax, nadam; ratio 6.2-13.8; lion and lamb have no opponent time). Open.
- moe synthetic (job afb26-moe, head eb3fca1ac): 72.7 ms, was 789 ms in refresh 7; ratio 1.93 vs torch-eager-bf16 37.7 ms. Still slower.
- Slower than the 0.8.34 FAST cells, still open: adafactor 184 -> 683 ms (flips slower), dynamic-optimized-theta taxi-hourly
  727 -> 2087 ms (flips slower), layernorm 26.8 -> 76.8 ms. Quality unchanged on these.
- Neural opponents now come from the M3 torch fill (opp3-neural-b, best of eager/compile x fp32/bf16): lstm-clf and
  lstm-reg trail torch at ratio 2.6-2.7; gru and rnn lanes stay ahead (0.53-0.77).
- Croston and GARCH register/grid defaults (main 2dcdd949f, job afb16-garch-croston): croston 2.9 -> 1.9 / 3.0 -> 2.3 ms,
  croston-sba 2.4 -> 2.2 / 3.0 -> 2.3 ms, croston-optimized 6.3 -> 9.4 / 9.9 -> 9.8 ms (synthetic slower, open); forecast_rmse
  unchanged.
- garch fixed in refresh 8 (job afb21-garch, head 5990c5946): 16.5 / 23.6 ms (synthetic / taxi-hourly), was 384 / 581 ms;
  ratio 0.16 / 0.19 vs arch-cpu (106 / 125 ms), faster than the 0.8.34 FAST cells (23.4 / 25.4 ms). mean_llf unchanged.
- Gram fast paths (LARS/RIDGE) + class-covariance grid (LDA/QDA, higher accuracy) default since c5e1bbeb6. LDA/QDA istella
  re-timed in refresh 8 (jobs afb20-eigh, afb-eighscope): lda-clf 19779 -> 533 ms, qda 15903 -> 429 ms; both flip faster
  (ratio 0.14 / 0.07 vs sklearn-cpu).
- lda-clf/qda taxi slightly slower than refresh 7 (30 -> 43, 116 -> 123 ms); both still faster than sklearn-cpu (ratio 0.23 / 0.87).
- IterativeImputer back on cyclic eigh under RR_EIGH (7edf6d895); LDA/QDA keep round-robin.
- DBSCAN on taxi times out in FAST (gap; lane apple-fast-dbscantaxi in progress).
- lda taxi-zones from A/B nb-lda-fused-zones, main a7b8b9513: 2724 -> 1509 ms (arm B median, digest 0d4f724587f2a9f7, perplexity 45.2218).
- layernorm synthetic from A/B regress-layernorm-pipe, main f6a9e7c04: 76.8 -> 52.2 ms (arm B median of 3, digest 535e09547fa4ea84 unchanged).
- adafactor synthetic from A/B regress-adafactor-pipe, main f6a9e7c04 (PIPE_UP+DOWN default): 683 -> 393 ms (arm B median of 3, digest 009c767eac9d71d0 unchanged).
- dynamic-optimized-theta taxi-hourly from A/B regress-dotm-snap, main f6a9e7c04: 2087 -> 562 ms (arm B median of 3, digest ba1f64384525a662 unchanged, forecast_rmse 49.0857).
- stl taxi-hourly from A/B tsa2-stl-taxi-hourly, main 550806bc0: 359 -> 7.0 ms (arm B, digest 5dd37db4cd817e98 -> 49e64d507212ec35, residual_std 18.3125 unchanged).
- select-r-regression taxi from A/B sel-rreg-taxi, main 4198d5a9c (SELECT_FREG default): new row, 103.5 (0.8.34 FAST) -> 10.1 ms (arm B, digest 3fb86b6cedc30f3d, n_selected 5).
- select-f-classif taxi from A/B sel-fcls-taxi, main e2bfb8422 (SELECT_FCLS default): new row, 128.2 (0.8.34 FAST) -> 17.5 ms (arm B, digest c1217e86bd2c518f, n_selected 5).
- select-f-regression taxi from A/B sel-freg-taxi-b, main e2bfb8422 (SELECT_FREG default since 4198d5a9c): new row, 102.8 (0.8.34 FAST) -> 15.0 ms (arm B, digest 14ef58d0e8f1bcee, n_selected 5).
- gbdt-depthwise taxi: fresh main time from gap A/B arm A, gapmisc-fusedq-dwtaxi, gapmisc-skipfs-dwtaxi, gapmisc-devscale-dwtaxi (arm A medians 10673 / 11108 / 10812): 11123 -> 10812 ms; logloss/auc = median of the 6 arm-A runs.
- autoarima taxi-hourly from A/B gaparima-async-taxi, main c73305a4c (ARIMA_FAST_ASYNC default): 39639 (fresh main, gap arm A) -> 24563 ms (arm B, digest 345eb06e29edb8f1 unchanged, forecast_rmse 74.6591).
- autoarima synthetic from A/B gaparima-async-synthetic, main c73305a4c (ARIMA_FAST_ASYNC default): 28578 (fresh main, gap arm A gaparima-noread-synthetic) -> 14918 ms (arm B, digest 9bdde6d5440d1cb6 unchanged, forecast_rmse 2.62412).
- knn istella (classical): fresh main time from gap A/B gapcls1-k64chk-istella arm B (defines '', KNN_FAST_MMA_K64 default): 1805 -> 357 ms (digest 61554d6d8e997190 -> 09e203c7fc8cf1a5, recall_at_k 0.9766 -> 0.9824).
- kpss taxi-hourly from A/B gaptsa-kpsspack-kpss-taxi-hourly, main 6c0ba4379 (gap-tsa merge): 3.8 -> 1.5 ms (arm B, digest 2202f943b403a767 -> d5e8380e1d592420, stationary_fraction 0.6875 unchanged).
- kpss synthetic from A/B gaptsa-kpsspack-kpss-synthetic, main 6c0ba4379 (gap-tsa merge): 3.9 -> 2.0 ms (arm B, digest f569a3ab20d3a4a0 -> ca25a7ac27f23f8a, stationary_fraction 0.03125 unchanged).
- select-d taxi-hourly from A/B gaptsa-seldfused-selectd-taxi-hourly, main 6c0ba4379 (gap-tsa merge): 3.9 (sel-d-hourly) -> 2.0 ms (arm B, digest a9b53cf12395d554 unchanged).
- select-d synthetic from A/B gaptsa-seldfused-selectd-synthetic, main 6c0ba4379 (gap-tsa merge): 6.2 -> 2.2 ms (arm B, digest 12c97f719e95f054 unchanged).
- var taxi-hourly from A/B gaptsa-varonecopy-var-taxi-hourly, main 6c0ba4379 (gap-tsa merge): 5.0 (tsa2-var-taxi-hourly) -> 5.1 ms (arm B, digest de5dcf6b776912c4 unchanged).
- var synthetic from A/B gaptsa-varonecopy-var-synthetic, main 6c0ba4379 (gap-tsa merge): 4.5 -> 5.0 ms (arm B, digest 1ffe3c6b4f2c3059 unchanged).
- theta taxi-hourly from A/B gaptsa-thetaspec-theta-taxi-hourly, main 6c0ba4379 (gap-tsa merge): 219 -> 20.3 ms (arm B, digest 51ee9c2838c11775 unchanged, forecast_rmse 49.0206).
- bayesian-ridge taxi from A/B gapcls1-brall-taxi, main a8e9ee5a3: 404 -> 15.3 ms (arm B; fresh main arm A median 101 over gapcls1-br*-taxi; digest -> 3ce1ea7b7da22638, r2 0.908983).
- ard taxi from A/B gapcls1-ardall-taxi, main a8e9ee5a3: 41.9 -> 8.3 ms (arm B; fresh main arm A median 14.3 over gapcls1-ard*-taxi; digest -> 1fe8d4121bf6e667, r2 0.90919).
- ridge-clf taxi from A/B gapcls1-rccodes-taxi, main a8e9ee5a3: 121 -> 19.0 ms (arm B; fresh main arm A median 120 over gapcls1-rc*-taxi; digest e4be71b46f63ea74 unchanged).
- nearest-centroid taxi from A/B gapcls1-nclabels-taxi, main a8e9ee5a3: 129 -> 20.9 ms (arm B; fresh main arm A median 106 over gapcls1-nc*-taxi; digest 92cb5f66b0f22a29 unchanged).
- gaussian-rp istella from A/B gapcls2-devscan-grp-istella, main 417ba1ded (cls2 merge): 54.3 (fresh main, gap arm A) -> 15.9 ms (arm B, digest ce7dcd780f8e2868 unchanged).
- gaussian-rp taxi from A/B gapcls2-devscan-grp-taxi, main 417ba1ded (cls2 merge): 4.5 (fresh main, gap arm A) -> 3.9 ms (arm B, digest 20cb8f6b620c564a unchanged).
- minmax-scaler istella from A/B gapcls2-fusedpool-minmax-istella, main 417ba1ded (cls2 merge): 106 (fresh main, gap arm A) -> 19.1 ms (arm B, digest 89e23b6ba666ae69 unchanged).
- minibatch-kmeans istella from A/B gapcls2-pool-mbk-istella, main 417ba1ded (cls2 merge): 256 (fresh main, gap arm A) -> 171 ms (arm B, digest 09785e3b5965a96b unchanged).
- minibatch-kmeans taxi from A/B gapcls2-pool-mbk-taxi, main 417ba1ded (cls2 merge): 42.5 (fresh main, gap arm A) -> 38.1 ms (arm B, digest 98e86aa3e8082355 unchanged).
- onehot taxi from A/B gapcls2-present-onehot-taxi, main 417ba1ded (cls2 merge): 28.0 -> 5.6 ms (arm B, digest f0cf38cb873b3cef unchanged).
- ordinal taxi from A/B gapcls2-present-ordinal-taxi, main 417ba1ded (cls2 merge): 24.8 -> 8.2 ms (arm B, digest 5055ebbcac96f05a unchanged).
- ocsvm taxi from A/B gapcls2-all3-ocsvm-taxi, main 417ba1ded (cls2 merge): 372 -> 65.0 ms (arm B, digest 1a7756c6125d21cc unchanged, fraction_flagged 0.1361).
- ivf-pq istella from A/B ann-ivfpq-devcb-istella-b, main 8d43ec357 (ANN merge): new row, 6215 (0.8.34 FAST) -> 2503 ms (arm B, digest be5c8cfcc9c64d4b, recall_at_10 0.7561).
- ivf-refine istella from A/B ann-ivfrefine-devcb-istella-b, main 8d43ec357 (ANN merge): new row, 6078 (0.8.34 FAST) -> 2507 ms (arm B, digest 001a81e6cb0cd5ee, recall_at_10 0.982).
- ivf-filter istella from A/B ann-ivffilter-devcb-istella-b, main 8d43ec357 (ANN merge): new row, 6190 (0.8.34 FAST) -> 2517 ms (arm B, digest 397630650b0c139a, recall_at_10 0.801).
- affinity-prop istella from A/B cluster2-ap-split-istella-b, main deee07721 (cluster2 merge): new row, 964 (0.8.34 FAST) -> 265 ms (arm B, digest 77287898046eb84c, n_clusters 342).
- optics istella from A/B cluster2-optics-devorder-istella-b, main deee07721 (cluster2 merge): new row, 514 (0.8.34 FAST) -> 261 ms (arm B, digest ed5d11e10edc67e0, n_clusters 20).
- stl synthetic: main-only timing stl-main-synthetic (main 413da4d18, single arm): 359 -> 7.27 ms (digest 3693977ba9768f2a, residual_std 0.783175 unchanged).
- adam / adamw synthetic: main-only timings nofast-adam-synthetic / nofast-adamw-synthetic (main 413da4d18): new rows, 343 / 338 ms (no prior M3 FAST time). Neural: not on the FAST page.
- als taxi-zones: main-only timing nofast-als-taxi-zones (main 413da4d18): new row, 17842 ms (no prior M3 FAST time; digest d1db7ca0daf40f2c, recall_at_10 0.054).
- als text, bayesian-gmm istella, bootstrap istella, bootstrap taxi: main-only timings nofast-als-text / nofast-bayesian-gmm-istella / nofast-bootstrap-istella / nofast-bootstrap-taxi (main 413da4d18, single arm): 127109 / 52250 / 14.2 / 9.4 ms (als text and bayesian-gmm istella new rows; bootstrap was 10.2 / 13.3 on the 0.8.34 board, digests unchanged quality).
- bayesian-ridge istella from A/B bayesq-istella, main 9525ed958 (BayesianRidge FAST quality fix): 2851 -> 494 ms (arm B, digest 360b6f8abeb08718); r2 -32316 -> 0.3284, rmse 150 -> 0.6846 (sklearn-cpu float32 r2 -890; sklearn float64 r2 0.3287). Counted again.
- cagra istella from A/B gapcagra-ivfgxsi-istella, main 8b9c91b9c (CAGRA IVFG+EXACTD+SEEDS+ITERS default): 21240 -> 1254 ms (arm B, digest 2c603ee8cee7c2ff); recall_at_10 0.9838 -> 0.9972 (faiss-cpu 0.9994).
- cagra taxi from A/B gapcagra-seedsiters-taxi, main 8b9c91b9c (CAGRA default): 2887 -> 2900 ms (arm B, digest 82ef82f92db5764b); recall_at_10 0.4838 -> 0.9979 (faiss-cpu 0.9277). Quality fixed; still slower than faiss-cpu.
- main-only timings on main c48ede3c1 (single arm, tags nofast-*-b and retime-*-b): cross-val-score, factor-analysis, gamma, knn-imputer, label-spreading, lof, logreg-cv, permutation-test, poisson, quantile, resample, sparse-pca, svgp, tweedie, voting-reg (nofast: no prior M3 FAST time); additive-chi2, bisecting-kmeans, categorical-nb (retime: replace stale 0.8.34 FAST cells).
- elliptic-envelope istella: nofast-elliptic-envelope-istella-b was killed (median none): FAST too slow to time; row stays without a FAST time.
- main-only retimes on main c48ede3c1 (single arm, retime-*-b): cholesky, complement-nb, connected-components, elliptic-envelope taxi, huber, incremental-pca taxi, isotonic, kernel-pca, kernel-shap istella, knn-imputer taxi, label-binarizer, label-encoder, lle, lstsq istella, lu-factor, lu-solve, maxabs-scaler, mb-dict-learning istella (replace stale 0.8.34 FAST cells).
- elliptic-envelope taxi: 78766 ms on main c48ede3c1 (retime-elliptic-envelope-taxi-b) vs the stale 1741 ms: regression on main; fix merging (batch robust-ee winner, 267 ms on the M2).
- **FAST quality under review:** knn-imputer taxi (masked_rmse 6.15 vs sklearn 5.26); not counted as faster until a quality lane clears it.
- kernel-shap istella: regression on main (14.4 s -> 27.0 s, retime-kernel-shap-istella-b on c48ede3c1); gap lane.
- gamma taxi, tweedie taxi: quality cleared by the quality lane: bench ill-posed; sklearn equally negative (gamma r2 -231.9 vs sklearn -233.0; tweedie -10.074 vs -10.074). Counted.
- **FAST quality under review (not counted as faster):** tweedie istella (r2 -24.4 vs sklearn -21.9; same in IDENTICAL; exp-link on unscaled features); factor-analysis istella (mean log-likelihood 89.0 vs sklearn 98.1; fix on lane/apple-fast-quality-glmfa, M3 tags qglm-factor-analysis-*).
