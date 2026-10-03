# M3 FAST board refresh (lane/apple-fast)

Our FAST arm on the M3 Ultra Metal GPU at head 0e743cac9, 24ed76679, 2dcdd949f, 37c65a3af, 42b6db46c, 4b1311c12, 69f7a41fd, 6c54b7e87, 6d4d55c99, 829c3fb4a, 94cb5ba59, 9722e5a2b, baa5d967a, c55b8c377, c5e1bbeb6, d1fa9223b, ea3c917de, ed49f2b11, f419ea9f1, 1 warm-up + 3 timed rounds at board size (rows-full; trees MOJOLEARN_SPEED_SIZE=shipped). Opponents are not re-raced: classical times come from the M3 0.8.34 board (`~/mojolearn-evidence/board-0834-times.tsv`, quality from its board.json), trees from the 2026-09-29 M3 board (older tree params on some lanes); an opponent marked (fill) comes from the M3 opponent fill on the 0.8.34 kit. Ratio = our FAST ms / best opponent ms; below 1 is faster. Rows sort worst ratio after first. Written by `tools/af_board_merge.py`.

Summary: 149 rows, 142 with a ratio, 97 faster than the best opponent after (bayesian-ridge istella excluded: NaN), geometric-mean ratio 0.53. Flips to faster: lasso-lars istella, onehot istella, ordinal istella, multioutput-reg taxi, iforest istella, qda taxi, dart-reg istella, dart istella, gbdt-categorical taxi, minmax-scaler taxi, lr-onecycle synthetic, nearest-centroid istella, enet-cv taxi, meanshift istella, bayesian-ridge istella, lasso-cv taxi, lasso-lars taxi, lars taxi, gbdt-lossguide istella, ridge-clf istella, lda taxi-zones, lasso-cv istella, damped-ets synthetic, damped-ets taxi-hourly, calibrated taxi, enet-cv istella, ard istella. Flips to slower: garch taxi-hourly, garch synthetic, adafactor synthetic, ridge taxi, dynamic-optimized-theta taxi-hourly.

| lane | dataset | family | FAST before ms | FAST after ms | best opponent | opp ms | ratio before | ratio after | flip | quality after (FAST) | quality before (FAST) | opponent quality | status |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| layernorm | synthetic | algos | 26.8 | 76.8 | torch-eager-bf16 | 2.8 | 9.57 | 27.41 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| adagrad | synthetic | algos | 131 | 324 | torch-eager-fp32 | 14.2 | 9.25 | 22.83 |  | - | - | - | ok |
| moe | synthetic | algos | 789 | 789 | torch-eager-bf16 | 37.7 | 20.92 | 20.92 |  | - | - | max_rel_diff_vs_torch_eager_fp32=2.234e+04, rel_fro_vs_torch_eager_fp32=0.05522 | ok |
| autoarima | taxi-hourly | algos | - | 39298 | statsforecast-cpu | 2844 | - | 13.82 |  | forecast_rmse=74.6591 | - | forecast_rmse=68.21 | ok |
| rmsprop | synthetic | algos | 129 | 318 | torch-eager-fp32 | 25.2 | 5.13 | 12.61 |  | - | - | - | ok |
| adamax | synthetic | algos | 176 | 325 | torch-eager-fp32 | 27.7 | 6.37 | 11.75 |  | - | - | - | ok |
| nadam | synthetic | algos | 178 | 331 | torch-eager-fp32 | 32.9 | 5.40 | 10.05 |  | - | - | - | ok |
| autoarima | synthetic | algos | - | 28504 | statsforecast-cpu | 2880 | - | 9.90 |  | forecast_rmse=2.62412 | - | forecast_rmse=17.55 | ok |
| bayesian-ridge | taxi | algos | 440 | 404 | sklearn-cpu | 73.2 | 6.01 | 5.52 |  | r2=0.908981, rmse=4.80511 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| lda-clf | istella | algos | 19228 | 19779 | sklearn-cpu | 3739 | 5.14 | 5.29 |  | accuracy=0.9131, logloss=0.235662 | accuracy=0.909, logloss=0.2644 | accuracy=0.9011, logloss=0.4492 | ok |
| garch | taxi-hourly | algos | 25.4 | 581 | arch-cpu | 125 | 0.20 | 4.66 | FLIP slower | mean_llf=-1132.96 | mean_llf=-1133 | mean_llf=-1130 | ok |
| garch | synthetic | algos | 23.4 | 384 | arch-cpu | 106 | 0.22 | 3.64 | FLIP slower | mean_llf=-1938.22 | mean_llf=-1938 | mean_llf=-1938 | ok |
| stl | synthetic | algos | 359 | 359 | statsmodels-cpu | 105 | 3.43 | 3.43 |  | residual_std=0.783175 | residual_std=0.7832 | residual_std=0.7832 | ok |
| knn | istella | classical | 1592 | 1805 | sklearn-cpu | 566 | 2.82 | 3.19 |  | recall_at_k=0.976613, rows_with_repeated_ids=0 | recall_at_k=0.9766, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| stl | taxi-hourly | algos | 359 | 359 | statsmodels-cpu | 119 | 3.02 | 3.02 |  | residual_std=18.3125 | residual_std=18.31 | residual_std=18.31 | ok |
| lr-exponential | synthetic | algos | 224 | 225 | torch-cpu | 76.7 | 2.92 | 2.93 |  | - | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.933e-08 | ok |
| minibatch-kmeans | istella | algos | 673 | 358 | sklearn-cpu | 124 | 5.45 | 2.90 |  | n_clusters=8, silhouette=0.118168 | ari_vs_ours=1, n_clusters=8, silhouette=0.1167 | ari_vs_ours=0.6224, n_clusters=8, silhouette=0.1119 | ok |
| lstm-reg | synthetic | algos | 1880 | 1878 | torch-compile-fp32 (fill) | 695 | 2.70 | 2.70 |  | r2=0.981013, rmse=0.159641 | r2=0.981, rmse=0.1596 | - | ok |
| lstm-reg | taxi-hourly | algos | 1878 | 1879 | torch-compile-fp32 (fill) | 700 | 2.68 | 2.68 |  | r2=0.751679, rmse=0.540429 | r2=0.7517, rmse=0.5404 | - | ok |
| lstm-clf | taxi-hourly | algos | 1879 | 1890 | torch-eager-fp32 (fill) | 729 | 2.58 | 2.59 |  | accuracy=0.868218, logloss=0.299901 | accuracy=0.8682, logloss=0.2999 | - | ok |
| lstm-clf | synthetic | algos | 1878 | 1882 | torch-eager-fp32 (fill) | 729 | 2.58 | 2.58 |  | accuracy=0.968696, logloss=0.0724409 | accuracy=0.9687, logloss=0.07244 | - | ok |
| qda | istella | algos | 16290 | 15903 | sklearn-cpu | 6397 | 2.55 | 2.49 |  | accuracy=0.8805, logloss=3.60845 | accuracy=0.8661, logloss=4.053 | accuracy=0.8805, logloss=3.477 | ok |
| ard | taxi | algos | 44.3 | 41.9 | sklearn-cpu | 17.5 | 2.53 | 2.40 |  | r2=0.909193, rmse=4.79951 | r2=0.9092, rmse=4.8 | r2=0.9092, rmse=4.8 | ok |
| gaussian-rp | istella | algos | 58.8 | 53.7 | sklearn-cpu | 24.3 | 2.42 | 2.21 |  | mean_abs_distortion=0.680693 | mean_abs_distortion=0.6807 | mean_abs_distortion=0.178 | ok |
| select-d | taxi-hourly | algos | 7.0 | 7.0 | statsmodels-cpu | 3.4 | 2.06 | 2.07 |  | - | - | d_agreement_vs_statsmodels=1 | ok |
| ocsvm | taxi | algos | 246 | 372 | sklearn-cpu | 181 | 1.36 | 2.06 |  | fraction_flagged=0.1361 | fraction_flagged=0.1361 | fraction_flagged=0.1361, jaccard_vs_sklearn=1 | ok |
| gaussian-rp | taxi | algos | 5.9 | 3.3 | sklearn-cpu | 1.6 | 3.69 | 2.06 |  | mean_abs_distortion=0.345752 | mean_abs_distortion=0.3458 | mean_abs_distortion=0.3398 | ok |
| adafactor | synthetic | algos | 184 | 683 | torch-eager-fp32 | 344 | 0.54 | 1.99 | FLIP slower | - | - | - | ok |
| var | taxi-hourly | algos | 14.5 | 5.1 | statsmodels-cpu | 2.8 | 5.18 | 1.82 |  | forecast_rmse=33.168 | forecast_rmse=33.17 | forecast_rmse=33.17 | ok |
| var | synthetic | algos | 15.9 | 4.5 | statsmodels-cpu | 2.6 | 6.12 | 1.73 |  | forecast_rmse=1.14079 | forecast_rmse=1.141 | forecast_rmse=1.145 | ok |
| minmax-scaler | istella | algos | 359 | 106 | sklearn-cpu | 63.0 | 5.70 | 1.69 |  | - | - | - | ok |
| ridge | taxi | classical2 | 31.5 | 55.7 | sklearn-cpu | 35.7 | 0.88 | 1.56 | FLIP slower | r2=0.908983, rmse=4.80505 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| ridge-clf | taxi | algos | 410 | 121 | sklearn-cpu | 77.9 | 5.27 | 1.56 |  | accuracy=0.76357 | accuracy=0.7636 | accuracy=0.7636 | ok |
| dynamic-optimized-theta | taxi-hourly | algos | 727 | 2087 | statsforecast-cpu | 1352 | 0.54 | 1.54 | FLIP slower | forecast_rmse=49.0857 | forecast_rmse=49.08 | forecast_rmse=49.31 | ok |
| kpss | synthetic | algos | 3.1 | 3.9 | statsmodels-cpu | 2.6 | 1.19 | 1.52 |  | stationary_fraction=0.03125 | stationary_fraction=0.03125 | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| onehot | taxi | algos | 71.8 | 28.0 | sklearn-cpu | 19.0 | 3.78 | 1.48 |  | - | - | - | ok |
| kpss | taxi-hourly | algos | 3.0 | 3.8 | statsmodels-cpu | 2.6 | 1.15 | 1.44 |  | stationary_fraction=0.6875 | stationary_fraction=0.6875 | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| nearest-centroid | taxi | algos | 471 | 129 | sklearn-cpu | 96.9 | 4.86 | 1.33 |  | accuracy=0.6667, logloss=0.781905 | accuracy=0.6667, logloss=0.7822 | accuracy=0.6667, logloss=0.7817 | ok |
| ordinal | taxi | algos | 67.2 | 24.8 | sklearn-cpu | 18.8 | 3.57 | 1.32 |  | - | - | - | ok |
| theta | taxi-hourly | algos | 1747 | 219 | statsmodels-cpu | 171 | 10.20 | 1.28 |  | forecast_rmse=49.0206 | forecast_rmse=49.28 | forecast_rmse=49.31 | ok |
| select-d | synthetic | algos | 6.6 | 6.2 | statsmodels-cpu | 5.1 | 1.29 | 1.22 |  | - | - | d_agreement_vs_statsmodels=1 | ok |
| minibatch-kmeans | taxi | algos | 106 | 51.7 | sklearn-cpu | 43.9 | 2.40 | 1.18 |  | n_clusters=8, silhouette=0.138023 | ari_vs_ours=1, n_clusters=8, silhouette=0.1381 | ari_vs_ours=0.5252, n_clusters=8, silhouette=0.1655 | ok |
| gbdt-depthwise | taxi | trees | 13994 | 11123 | xgboost-cpu (fill) | 10435 | 1.34 | 1.07 |  | logloss=0.527847, auc=0.63232 | auc=0.6258, logloss=0.53 | auc=0.631 | ok |
| knn-clf | istella | classical2 | 834 | 308 | sklearn-cpu | 292 | 2.85 | 1.05 |  | accuracy=0.92625 | accuracy=0.9263 | accuracy=0.9263 | ok |
| lasso-lars | istella | algos | 1016 | 227 | sklearn-cpu | 228 | 4.45 | 1.00 | FLIP faster | r2=0.310837, rmse=0.69346 | r2=0.3103, rmse=0.6937 | r2=0.3111, rmse=0.6933 | ok |
| onehot | istella | algos | 78.8 | 36.5 | sklearn-cpu | 36.7 | 2.15 | 0.99 | FLIP faster | - | - | - | ok |
| ordinal | istella | algos | 79.8 | 36.2 | sklearn-cpu | 36.9 | 2.16 | 0.98 | FLIP faster | - | - | - | ok |
| multioutput-reg | taxi | algos | 115 | 50.2 | sklearn-cpu | 53.7 | 2.14 | 0.94 | FLIP faster | r2=0.60424 | r2=0.6042 | r2=0.6043 | ok |
| iforest | istella | trees | 437 | 262 | sklearn-iforest-cpu (fill) | 304 | 1.44 | 0.86 | FLIP faster | auc=0.830358 | auc=0.8304 | auc=0.8279 | ok |
| qda | taxi | algos | 187 | 116 | sklearn-cpu | 141 | 1.33 | 0.82 | FLIP faster | accuracy=0.72722, logloss=1.05927 | accuracy=0.727, logloss=1.061 | accuracy=0.7272, logloss=1.059 | ok |
| mlp-reg | taxi | algos | 8033 | 8031 | sklearn-cpu | 9959 | 0.81 | 0.81 |  | r2=0.931981, rmse=4.15387 | r2=0.932, rmse=4.154 | r2=0.9296, rmse=4.226 | ok |
| mlp-reg | istella | algos | 10761 | 10761 | sklearn-cpu | 13651 | 0.79 | 0.79 |  | r2=0.526405, rmse=0.574862 | r2=0.5273, rmse=0.5743 | r2=0.5248, rmse=0.5758 | ok |
| rnn-reg | synthetic | algos | 1497 | 1564 | torch-eager-fp32 (fill) | 2037 | 0.73 | 0.77 |  | r2=0.977348, rmse=0.174374 | r2=0.9774, rmse=0.1744 | - | ok |
| rnn-reg | taxi-hourly | algos | 1572 | 1560 | torch-eager-fp32 (fill) | 2049 | 0.77 | 0.76 |  | r2=0.738796, rmse=0.554271 | r2=0.7388, rmse=0.5543 | - | ok |
| mlp-clf | istella | algos | 10766 | 10773 | sklearn-cpu | 14327 | 0.75 | 0.75 |  | accuracy=0.94435, logloss=0.136481 | accuracy=0.9446, logloss=0.1364 | accuracy=0.9438, logloss=0.1368 | ok |
| auto-theta | taxi-hourly | algos | 3433 | 2910 | statsforecast-cpu | 4032 | 0.85 | 0.72 |  | forecast_rmse=49.0542 | forecast_rmse=49.31 | forecast_rmse=49.27 | ok |
| dart-reg | istella | algos | 69437 | 23099 | lightgbm-cpu | 32389 | 2.14 | 0.71 | FLIP faster | r2=0.551355, rmse=0.559515 | r2=0.5507, rmse=0.5599 | r2=0.5647, rmse=0.5512 | ok |
| gbdt-depthwise | istella | trees | 19062 | 16551 | xgboost-cpu (fill) | 23218 | 0.82 | 0.71 |  | logloss=0.156903, auc=0.98319 | auc=0.9802, logloss=0.1819 | auc=0.9836 | ok |
| rnn-clf | taxi-hourly | algos | 1485 | 1484 | torch-eager-fp32 (fill) | 2087 | 0.71 | 0.71 |  | accuracy=0.868056, logloss=0.304864 | accuracy=0.8681, logloss=0.3049 | - | ok |
| rnn-clf | synthetic | algos | 1568 | 1483 | torch-eager-fp32 (fill) | 2088 | 0.75 | 0.71 |  | accuracy=0.953559, logloss=0.103698 | accuracy=0.9536, logloss=0.1037 | - | ok |
| mlp-clf | taxi | algos | 8052 | 8050 | sklearn-cpu | 11431 | 0.70 | 0.70 |  | accuracy=0.76783, logloss=0.530453 | accuracy=0.7678, logloss=0.5305 | accuracy=0.7678, logloss=0.5304 | ok |
| theta | synthetic | algos | 99.6 | 72.0 | statsmodels-cpu | 104 | 0.96 | 0.69 |  | forecast_rmse=1.43661 | forecast_rmse=1.437 | forecast_rmse=1.435 | ok |
| ocsvm | istella | algos | 446 | 451 | sklearn-cpu | 691 | 0.65 | 0.65 |  | fraction_flagged=0.0783 | fraction_flagged=0.0783 | fraction_flagged=0.0783, jaccard_vs_sklearn=1 | ok |
| dart | istella | algos | 69424 | 23014 | lightgbm-cpu | 35425 | 1.96 | 0.65 | FLIP faster | accuracy=0.9486, logloss=0.134461 | accuracy=0.9487, logloss=0.1341 | accuracy=0.9519, logloss=0.1237 | ok |
| knn | taxi | classical | 224 | 257 | sklearn-cpu | 426 | 0.53 | 0.60 |  | recall_at_k=0.999754, rows_with_repeated_ids=0 | recall_at_k=0.9998, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| gbdt-ordered | taxi | trees | - | 56105 | catboost-cpu (fill) | 99253 | - | 0.57 |  | logloss=0.52928, auc=0.628294 | - | auc=0.6277 | ok |
| gbdt-categorical | taxi | trees | 63398 | 32925 | lightgbm-cpu (fill) | 58904 | 1.08 | 0.56 | FLIP faster | logloss=0.528426, auc=0.63103 | auc=0.6302, logloss=0.5286 | auc=0.6327 | ok |
| gru-clf | taxi-hourly | algos | 1632 | 1650 | torch-eager-fp32 (fill) | 3099 | 0.53 | 0.53 |  | accuracy=0.865668, logloss=0.305841 | accuracy=0.8657, logloss=0.3058 | - | ok |
| gru-reg | synthetic | algos | 1630 | 1629 | torch-eager-fp32 (fill) | 3060 | 0.53 | 0.53 |  | r2=0.981946, rmse=0.155672 | r2=0.9819, rmse=0.1557 | - | ok |
| gru-reg | taxi-hourly | algos | 1643 | 1627 | torch-eager-fp32 (fill) | 3072 | 0.54 | 0.53 |  | r2=0.748219, rmse=0.544182 | r2=0.7482, rmse=0.5442 | - | ok |
| gru-clf | synthetic | algos | 1645 | 1653 | torch-eager-fp32 (fill) | 3137 | 0.52 | 0.53 |  | accuracy=0.971842, logloss=0.0658348 | accuracy=0.9718, logloss=0.06584 | - | ok |
| minmax-scaler | taxi | algos | 22.1 | 9.0 | sklearn-cpu | 17.6 | 1.26 | 0.51 | FLIP faster | - | - | - | ok |
| lr-onecycle | synthetic | algos | 1359 | 55.6 | torch-cpu | 111 | 12.26 | 0.50 | FLIP faster | - | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.951e-08 | ok |
| nearest-centroid | istella | algos | 2896 | 263 | sklearn-cpu | 529 | 5.47 | 0.50 | FLIP faster | accuracy=0.85261, logloss=4.29948 | accuracy=0.8526, logloss=4.299 | accuracy=0.8526, logloss=4.118 | ok |
| enet-cv | taxi | algos | 3173 | 105 | sklearn-cpu | 211 | 15.03 | 0.50 | FLIP faster | r2=0.909004, rmse=4.80449 | r2=0.909, rmse=4.804 | r2=0.909, rmse=4.804 | ok |
| meanshift | istella | algos | 5834 | 214 | sklearn-cpu | 444 | 13.13 | 0.48 | FLIP faster | n_clusters=12, silhouette=0.403452 | ari_vs_ours=1, n_clusters=12, silhouette=0.4034 | ari_vs_ours=1, n_clusters=12, silhouette=0.4034 | ok |
| lda | text | algos | - | 34671 | sklearn-cpu | 72657 | - | 0.48 |  | perplexity=266.719 | - | perplexity=266.9 | ok |
| bayesian-ridge | istella | algos | 92368 | 3163 | sklearn-cpu | 6831 | 13.52 | 0.46 | FLIP faster | r2=nan, rmse=nan | r2=-4.169e+04, rmse=170.6 | r2=-890.2, rmse=24.94 | ok |
| lasso-cv | taxi | algos | 3170 | 104 | sklearn-cpu | 226 | 14.00 | 0.46 | FLIP faster | r2=0.909038, rmse=4.80359 | r2=0.9091, rmse=4.803 | r2=0.909, rmse=4.804 | ok |
| knn-clf | taxi | classical2 | 48.5 | 65.3 | sklearn-cpu | 154 | 0.32 | 0.42 |  | accuracy=0.74175 | accuracy=0.7418 | accuracy=0.7418 | ok |
| gbdt-symmetric-1000 | taxi | trees | 19086 | 22353 | catboost-cpu (fill) | 55261 | 0.35 | 0.40 |  | logloss=0.528216, auc=0.631675 | auc=0.6239, logloss=0.5306 | auc=0.6316 | ok |
| gbdt-symmetric | taxi | trees | 9520 | 11253 | catboost-cpu (fill) | 28188 | 0.34 | 0.40 |  | logloss=0.528595, auc=0.630376 | auc=0.6239, logloss=0.5306 | auc=0.6303 | ok |
| dynamic-theta | taxi-hourly | algos | 396 | 303 | statsforecast-cpu | 768 | 0.51 | 0.39 |  | forecast_rmse=49.1014 | forecast_rmse=49.1 | forecast_rmse=49.27 | ok |
| dynamic-optimized-theta | synthetic | algos | 485 | 377 | statsforecast-cpu | 959 | 0.51 | 0.39 |  | forecast_rmse=1.43637 | forecast_rmse=1.436 | forecast_rmse=1.436 | ok |
| ols | taxi | classical | 87.3 | 106 | sklearn-cpu | 274 | 0.32 | 0.39 |  | r2=0.908838, rmse=4.69644 | r2=0.9088, rmse=4.696 | r2=0.7248, rmse=8.159 | ok |
| lasso-lars | taxi | algos | 184 | 12.8 | sklearn-cpu | 34.6 | 5.32 | 0.37 | FLIP faster | r2=0.908998, rmse=4.80466 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.804 | ok |
| auto-theta | synthetic | algos | 962 | 718 | statsforecast-cpu | 1953 | 0.49 | 0.37 |  | forecast_rmse=1.43688 | forecast_rmse=1.438 | forecast_rmse=1.438 | ok |
| gbdt-rank-pairlogit | istella | trees | 3997 | 3064 | xgboost-cpu (fill) | 8355 | 0.48 | 0.37 |  | ndcg10=0.719953, ndcg5=0.6504, map=0.854545 | map=0.8414, ndcg10=0.7093, ndcg5=0.6398 | map=0.8728 | ok |
| lars | taxi | algos | 185 | 12.1 | sklearn-cpu | 34.2 | 5.40 | 0.35 | FLIP faster | r2=0.908983, rmse=4.80505 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| gbdt-lossguide | istella | trees | 106296 | 18364 | lightgbm-cpu (fill) | 55012 | 1.93 | 0.33 | FLIP faster | logloss=0.148902, auc=0.98367 | auc=0.9838, logloss=0.1488 | auc=0.9838 | ok |
| gbdt-multiclass | taxi | trees | 14685 | 14938 | xgboost-cpu (fill) | 45500 | 0.32 | 0.33 |  | mlogloss=1.01259, accuracy=0.59938 | accuracy=0.5967, mlogloss=1.023 | accuracy=0.6011 | ok |
| optimized-theta | taxi-hourly | algos | 576 | 398 | statsforecast-cpu | 1290 | 0.45 | 0.31 |  | forecast_rmse=49.1549 | forecast_rmse=49.15 | forecast_rmse=49.36 | ok |
| ridge-clf | istella | algos | 9790 | 2153 | sklearn-cpu | 7044 | 1.39 | 0.31 | FLIP faster | accuracy=0.91054 | accuracy=0.8943 | accuracy=0.9105 | ok |
| optimized-theta | synthetic | algos | 311 | 227 | statsforecast-cpu | 748 | 0.42 | 0.30 |  | forecast_rmse=1.43851 | forecast_rmse=1.44 | forecast_rmse=1.438 | ok |
| gbdt-symmetric | istella | trees | 11943 | 16955 | catboost-cpu (fill) | 60136 | 0.20 | 0.28 |  | logloss=0.186714, auc=0.980109 | auc=0.9756, logloss=0.2112 | auc=0.9799 | ok |
| gbdt-symmetric-1000 | istella | trees | 21364 | 32889 | catboost-cpu (fill) | 120805 | 0.18 | 0.27 |  | logloss=0.1704, auc=0.982434 | auc=0.9756, logloss=0.2112 | auc=0.9823 | ok |
| ols | istella | classical | 1332 | 850 | sklearn-cpu | 3245 | 0.41 | 0.26 |  | r2=0.331943, rmse=0.682027 | r2=0.3211, rmse=0.6875 | r2=0.001881, rmse=0.8337 | ok |
| dynamic-theta | synthetic | algos | 116 | 85.9 | statsforecast-cpu | 333 | 0.35 | 0.26 |  | forecast_rmse=1.43726 | forecast_rmse=1.437 | forecast_rmse=1.437 | ok |
| gbdt-multiclass | istella | trees | 21754 | 25728 | xgboost-cpu | 100166 | 0.22 | 0.26 |  | mlogloss=0.258413, accuracy=0.907556 | accuracy=0.9033, mlogloss=0.2809 | accuracy=0.9101, mlogloss=0.2468 | ok |
| gbdt-lossguide | taxi | trees | 44556 | 12721 | lightgbm-cpu (fill) | 51864 | 0.86 | 0.25 |  | logloss=0.528132, auc=0.631865 | auc=0.631, logloss=0.5283 | auc=0.6322 | ok |
| ovr | taxi | algos | 458 | 324 | sklearn-cpu | 1461 | 0.31 | 0.22 |  | accuracy=0.47894 | accuracy=0.4789 | accuracy=0.4789 | ok |
| arima | synthetic | classical2 | 108 | 80.5 | statsmodels-cpu | 364 | 0.30 | 0.22 |  | forecast_rmse=1.51554, insample_rmse=0.999341, mean_aic=5680.97, mean_llf=-2836.49 | forecast_rmse=1.516, insample_rmse=0.9993, mean_aic=5681, mean_llf=-2836 | forecast_rmse=1.515, insample_rmse=0.9993, mean_aic=5681, mean_llf=-2836 | ok |
| ridge | istella | classical2 | 879 | 1417 | sklearn-cpu | 6717 | 0.13 | 0.21 |  | r2=0.328682, rmse=0.684423 | r2=0.3205, rmse=0.6886 | r2=0.3287, rmse=0.6844 | ok |
| dart | taxi | algos | 10520 | 5378 | lightgbm-cpu | 26498 | 0.40 | 0.20 |  | accuracy=0.76815, logloss=0.529149 | accuracy=0.7683, logloss=0.5291 | accuracy=0.7682, logloss=0.5291 | ok |
| rf | taxi | trees | 10796 | 10708 | lightgbm-cpu | 53864 | 0.20 | 0.20 |  | logloss=0.525953, auc=0.617838 | auc=0.6178, logloss=0.526 | auc=0.617, logloss=0.5264 | ok |
| dart-reg | taxi | algos | 9932 | 5199 | lightgbm-cpu | 27650 | 0.36 | 0.19 |  | r2=0.925497, rmse=4.34734 | r2=0.9255, rmse=4.347 | r2=0.9263, rmse=4.325 | ok |
| lda-clf | taxi | algos | 87.4 | 30.1 | sklearn-cpu | 186 | 0.47 | 0.16 |  | accuracy=0.76253, logloss=0.539763 | accuracy=0.7626, logloss=0.5397 | accuracy=0.7625, logloss=0.5398 | ok |
| croston-optimized | taxi-hourly | algos | 10.4 | 9.8 | statsforecast-cpu | 60.5 | 0.17 | 0.16 |  | forecast_rmse=1.39825 | forecast_rmse=1.398 | forecast_rmse=1.398 | ok |
| adaboost-reg | taxi | algos | 2609 | 1887 | sklearn-cpu | 12197 | 0.21 | 0.15 |  | r2=-0.605962, rmse=20.1839 | r2=0.2164, rmse=14.1 | r2=0.5639, rmse=10.52 | ok |
| rf | istella | trees | 13810 | 10162 | lightgbm-cpu | 67210 | 0.21 | 0.15 |  | logloss=0.182017, auc=0.945385 | auc=0.9454, logloss=0.182 | auc=0.9454, logloss=0.1954 | ok |
| croston-optimized | synthetic | algos | 5.9 | 9.4 | statsforecast-cpu | 62.3 | 0.09 | 0.15 |  | forecast_rmse=1.67554 | forecast_rmse=1.675 | forecast_rmse=1.675 | ok |
| iforest | taxi | trees | 119 | 85.8 | sklearn-iforest-cpu (fill) | 575 | 0.21 | 0.15 |  | auc=0.551846 | auc=0.5518 | auc=0.5528 | ok |
| stacking-reg | taxi | algos | - | 853 | sklearn-cpu | 6008 | - | 0.14 |  | r2=0.919717, rmse=4.51282 | - | r2=0.9325, rmse=4.137 | ok |
| lr-step | synthetic | algos | 11.7 | 11.9 | torch-cpu | 85.1 | 0.14 | 0.14 |  | - | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=1.49e-08 | ok |
| et | taxi | trees | 3376 | 2854 | sklearn-et-cpu | 20629 | 0.16 | 0.14 |  | logloss=0.526142, auc=0.618907 | auc=0.6189, logloss=0.5261 | auc=0.619, logloss=0.526 | ok |
| et | istella | trees | 4360 | 4034 | sklearn-et-cpu | 30617 | 0.14 | 0.13 |  | logloss=0.189989, auc=0.937987 | auc=0.938, logloss=0.19 | auc=0.9379, logloss=0.1901 | ok |
| lda | taxi-zones | algos | 41177 | 2724 | sklearn-cpu | 20926 | 1.97 | 0.13 | FLIP faster | perplexity=45.2218 | perplexity=45.22 | perplexity=44.9 | ok |
| multioutput-clf | taxi | algos | 228 | 87.7 | sklearn-cpu | 686 | 0.33 | 0.13 |  | accuracy=0.86356 | accuracy=0.8636 | accuracy=0.8636 | ok |
| ridge-cv | istella | algos | - | 14955 | sklearn-cpu | 134463 | - | 0.11 |  | r2=0.328684, rmse=0.684422 | - | r2=0.3287, rmse=0.6844 | ok |
| multioutput-reg | istella | algos | 1823 | 1419 | sklearn-cpu | 13402 | 0.14 | 0.11 |  | r2=0.455327 | r2=0.4399 | r2=0.4553 | ok |
| lasso-cv | istella | algos | 70206 | 556 | sklearn-cpu | 5483 | 12.80 | 0.10 | FLIP faster | r2=0.325504, rmse=0.686041 | r2=0.3103, rmse=0.6937 | r2=0.3108, rmse=0.6935 | ok |
| prophet | synthetic | algos | 70.6 | 44.8 | prophet-cpu | 452 | 0.16 | 0.10 |  | forecast_rmse=1.01491 | forecast_rmse=1.015 | forecast_rmse=1.015 | ok |
| stacking-clf | taxi | algos | 1080 | 702 | sklearn-cpu | 7095 | 0.15 | 0.10 |  | accuracy=0.76792, logloss=0.536364 | accuracy=0.7679, logloss=0.5364 | accuracy=0.7553, logloss=0.5478 | ok |
| damped-ets | synthetic | algos | 530 | 16.1 | statsforecast-cpu | 167 | 3.17 | 0.10 | FLIP faster | forecast_rmse=13.9307 | forecast_rmse=13.93 | forecast_rmse=13.95 | ok |
| damped-ets | taxi-hourly | algos | 612 | 17.3 | statsforecast-cpu | 185 | 3.31 | 0.09 | FLIP faster | forecast_rmse=96.6846 | forecast_rmse=96.69 | forecast_rmse=96.69 | ok |
| multioutput-clf | istella | algos | 2212 | 2087 | sklearn-cpu | 22340 | 0.10 | 0.09 |  | accuracy=0.959175 | accuracy=0.9592 | accuracy=0.9592 | ok |
| calibrated | taxi | algos | 1330 | 48.8 | sklearn-cpu | 556 | 2.39 | 0.09 | FLIP faster | accuracy=0.75533, logloss=0.550718 | accuracy=0.7553, logloss=0.5507 | accuracy=0.7553, logloss=0.5507 | ok |
| enet-cv | istella | algos | 74148 | 581 | sklearn-cpu | 6979 | 10.62 | 0.08 | FLIP faster | r2=0.326794, rmse=0.685384 | r2=0.3166, rmse=0.6906 | r2=0.3173, rmse=0.6902 | ok |
| ard | istella | algos | 47943 | 863 | sklearn-cpu | 10421 | 4.60 | 0.08 | FLIP faster | r2=-0.138726, rmse=0.891395 | r2=-0.1235, rmse=0.8854 | r2=0.3274, rmse=0.6851 | ok |
| prophet | taxi-hourly | algos | 72.7 | 43.2 | prophet-cpu | 549 | 0.13 | 0.08 |  | forecast_rmse=32.0407 | forecast_rmse=32.02 | forecast_rmse=32.03 | ok |
| calibrated | istella | algos | 2291 | 193 | sklearn-cpu | 2784 | 0.82 | 0.07 |  | accuracy=0.88509, logloss=0.289824 | accuracy=0.8851, logloss=0.2898 | accuracy=0.8851, logloss=0.2898 | ok |
| ovr | istella | algos | 5375 | 5226 | sklearn-cpu | 75802 | 0.07 | 0.07 |  | accuracy=0.8927 | accuracy=0.8927 | accuracy=0.8927 | ok |
| stacking-reg | istella | algos | - | 10306 | sklearn-cpu | 153421 | - | 0.07 |  | r2=0.448252, rmse=0.620485 | - | r2=0.4476, rmse=0.6208 | ok |
| croston-sba | synthetic | algos | 2.9 | 2.2 | statsforecast-cpu | 34.5 | 0.08 | 0.06 |  | forecast_rmse=1.67446 | forecast_rmse=1.675 | forecast_rmse=1.675 | ok |
| stacking-clf | istella | algos | 4053 | 9413 | sklearn-cpu | 149799 | 0.03 | 0.06 |  | accuracy=0.92997, logloss=0.193693 | accuracy=0.93, logloss=0.1937 | accuracy=0.93, logloss=0.1939 | ok |
| adaboost-clf | taxi | algos | 3118 | 1616 | sklearn-cpu | 27304 | 0.11 | 0.06 |  | accuracy=0.76523, logloss=0.543302 | accuracy=0.7652, logloss=0.5432 | accuracy=0.7654, logloss=0.5406 | ok |
| croston-sba | taxi-hourly | algos | 2.5 | 2.3 | statsforecast-cpu | 47.9 | 0.05 | 0.05 |  | forecast_rmse=1.38668 | forecast_rmse=1.387 | forecast_rmse=1.387 | ok |
| croston | taxi-hourly | algos | 3.1 | 2.3 | statsforecast-cpu | 54.2 | 0.06 | 0.04 |  | forecast_rmse=1.39026 | forecast_rmse=1.39 | forecast_rmse=1.39 | ok |
| ridge-cv | taxi | algos | - | 33.1 | sklearn-cpu | 946 | - | 0.04 |  | r2=0.908983, rmse=4.80505 | - | r2=0.909, rmse=4.805 | ok |
| croston | synthetic | algos | 2.9 | 1.9 | statsforecast-cpu | 91.7 | 0.03 | 0.02 |  | forecast_rmse=1.67484 | forecast_rmse=1.675 | forecast_rmse=1.675 | ok |
| adaboost-reg | istella | algos | 6071 | 1820 | sklearn-cpu | 183985 | 0.03 | 0.01 |  | r2=0.242748, rmse=0.72691 | r2=0.2389, rmse=0.7288 | r2=0.1676, rmse=0.7621 | ok |
| meanshift | taxi | algos | 538 | 43.6 | sklearn-cpu | 9370 | 0.06 | 0.00 |  | n_clusters=122, silhouette=0.246631 | ari_vs_ours=1, n_clusters=122, silhouette=0.2466 | ari_vs_ours=1, n_clusters=122, silhouette=0.2466 | ok |
| adaboost-clf | istella | algos | 15452 | 4763 | - | - | - | - |  | accuracy=0.93715, logloss=0.431913 | accuracy=0.935, logloss=0.4387 | - | ok |
| gbdt-categorical | taxicat | trees | - | 37105 | - | - | - | - |  | logloss=0.528399, auc=0.630905 | - | - | ok |
| gbdt-ordered | istella | trees | - | 75359 | - | - | - | - |  | logloss=0.190603, auc=0.979518 | - | - | ok |
| gbdt-rank-yetirank | istella | trees | 25773 | - | lightgbm-cpu (fill) | 7083 | 3.64 | - |  | - | map=0.8149, ndcg10=0.681, ndcg5=0.6151 | map=0.8584 | refused |
| lamb | synthetic | algos | - | 339 | - | - | - | - |  | - | - | - | ok |
| lars | istella | algos | 5583 | 845 | - | - | - | - |  | r2=0.328662, rmse=0.684433 | r2=0.2157, rmse=0.7398 | - | ok |
| lion | synthetic | algos | 137 | 321 | - | - | - | - |  | - | - | - | ok |

Sources: before = M3 0.8.34 board (classical), M3 2026-09-29 board FAST cells (trees); job tags afb-moefix, afb-optfix, afb10-ols, afb11-arima-c2, afb12-seq-algos-1, afb12-seq-algos-2, afb12-seq-algos-3, afb12-seq-algos-4, afb13-forest, afb14-lossguide, afb15-prophet, afb16-garch-croston, afb17-gram-algos, afb17-gram-classical2, afb18-cluster, afb19-prep, afb2-algos-2, afb2-classical-4, afb2-classical2-3, afb2-trees-1b, afb3-cat-taxicat, afb3-trees-fast, afb4-grp-ocsvm-main, afb5-merged-algos, afb5-merged-trees, afb5b-dart, afb6-te-algos, afb9-dw2.

## Quality flags (M3 manager, 2026-10-02, updated 2026-10-03 refresh 7)

- **bayesian-ridge istella: NaN in this table.** The time comes from a run before the guard fix: the Gram sse shortcut
  cancels in f32 on istella. The guard (`MOJOLEARN_BAYES_GRID_GUARD`, default on Apple) is merged in main 6d4d55c99
  (istella NaN -> finite, job bayes-br-guard-istella); the board re-time is owed. Not counted as a win.
- **ard istella: r2 -0.139 vs sklearn 0.327.** FAST was already below sklearn before this pass (-0.124); the row-pass arm
  gives -0.122. Speed is real, quality is not at the opponent's level: open.
- **gbdt-rank-yetirank:** the refresh asked for dataset istella; the lane races on istellarank, so the driver refused.
  YetiRank's M3 FAST time on istellarank is 5.6 s (LEDGER), vs LightGBM 6.8 s on the board.
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
- Optimizers re-timed after the handle fix (main 9a7564bc5, job afb-optfix): adamax, adagrad, rmsprop, nadam, lion, lamb
  FAST 318-339 ms, slower than 0.8.34 FAST (129-178 ms) and far behind torch-eager-fp32 (14-33 ms on adagrad, rmsprop, adamax, nadam; lion and lamb have no opponent time).
- moe synthetic fixed (job afb-moefix, head baa5d967a): 789 ms, was refusing; ratio 20.9 vs torch-eager-bf16 37.7 ms.
- Slower than before within refresh 4: garch 23-25 -> 384-581 ms (flips slower), adafactor 184 -> 683 ms (flips slower),
  dynamic-optimized-theta taxi-hourly 727 -> 2087 ms (flips slower), layernorm 26.8 -> 76.8 ms. Quality unchanged on these.
- Neural opponents now come from the M3 torch fill (opp3-neural-b, best of eager/compile x fp32/bf16): lstm-clf and
  lstm-reg trail torch at ratio 2.6-2.7; gru and rnn lanes stay ahead (0.53-0.77).
- Croston and GARCH register/grid defaults (main 2dcdd949f, job afb16-garch-croston): croston 2.9 -> 1.9 / 3.0 -> 2.3 ms,
  croston-sba 2.4 -> 2.2 / 3.0 -> 2.3 ms, croston-optimized 6.3 -> 9.4 / 9.9 -> 9.8 ms (synthetic slower, open); forecast_rmse
  unchanged. garch 384 / 581 ms with the same digests as afb12-seq-algos-2: the new defaults did not move the board's garch
  time, still slower than arch-cpu (106 / 125 ms). Open.
- Gram fast paths (LARS/RIDGE) + class-covariance grid (LDA/QDA, higher accuracy) default since c5e1bbeb6; LDA/QDA istella speed lane in progress.
