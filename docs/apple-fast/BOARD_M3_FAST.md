# M3 FAST board refresh (lane/apple-fast)

Our FAST arm on the M3 Ultra Metal GPU at head 0e743cac9, 24ed76679, 37c65a3af, 42b6db46c, 6d4d55c99, 829c3fb4a, ed49f2b11, 1 warm-up + 3 timed rounds at board size (rows-full; trees MOJOLEARN_SPEED_SIZE=shipped). Opponents are not re-raced: classical times come from the M3 0.8.34 board (`~/mojolearn-evidence/board-0834-times.tsv`, quality from its board.json), trees from the 2026-09-29 M3 board (older tree params on some lanes); an opponent marked (fill) comes from the M3 opponent fill on the 0.8.34 kit. Ratio = our FAST ms / best opponent ms; below 1 is faster. Rows sort worst ratio after first. Written by `tools/af_board_merge.py`.

Summary: 50 rows, 47 with a ratio, 37 faster than the best opponent after (bayesian-ridge istella excluded: NaN), geometric-mean ratio 0.41. Flips to faster: multioutput-reg taxi, iforest istella, dart-reg istella, dart istella, gbdt-categorical taxi, nearest-centroid istella, enet-cv taxi, bayesian-ridge istella, lasso-cv taxi, gbdt-lossguide istella, lasso-cv istella, damped-ets taxi-hourly, calibrated taxi, damped-ets synthetic, enet-cv istella, ard istella. Flips to slower: none.

| lane | dataset | family | FAST before ms | FAST after ms | best opponent | opp ms | ratio before | ratio after | flip | quality after (FAST) | quality before (FAST) | opponent quality | status |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| bayesian-ridge | taxi | algos | 440 | 404 | sklearn-cpu | 73.2 | 6.01 | 5.52 |  | r2=0.908981, rmse=4.80511 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| knn | istella | classical | 1592 | 1805 | sklearn-cpu | 566 | 2.82 | 3.19 |  | recall_at_k=0.976613, rows_with_repeated_ids=0 | recall_at_k=0.9766, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| ard | taxi | algos | 44.3 | 41.9 | sklearn-cpu | 17.5 | 2.53 | 2.40 |  | r2=0.909193, rmse=4.79951 | r2=0.9092, rmse=4.8 | r2=0.9092, rmse=4.8 | ok |
| gaussian-rp | istella | algos | 58.8 | 53.7 | sklearn-cpu | 24.3 | 2.42 | 2.21 |  | mean_abs_distortion=0.680693 | mean_abs_distortion=0.6807 | mean_abs_distortion=0.178 | ok |
| ocsvm | taxi | algos | 246 | 372 | sklearn-cpu | 181 | 1.36 | 2.06 |  | fraction_flagged=0.1361 | fraction_flagged=0.1361 | fraction_flagged=0.1361, jaccard_vs_sklearn=1 | ok |
| gaussian-rp | taxi | algos | 5.9 | 3.3 | sklearn-cpu | 1.6 | 3.69 | 2.06 |  | mean_abs_distortion=0.345752 | mean_abs_distortion=0.3458 | mean_abs_distortion=0.3398 | ok |
| nearest-centroid | taxi | algos | 471 | 129 | sklearn-cpu | 96.9 | 4.86 | 1.33 |  | accuracy=0.6667, logloss=0.781905 | accuracy=0.6667, logloss=0.7822 | accuracy=0.6667, logloss=0.7817 | ok |
| gbdt-depthwise | taxi | trees | 13994 | 13144 | xgboost-cpu (fill) | 10435 | 1.34 | 1.26 |  | logloss=0.527881, auc=0.632264 | auc=0.6258, logloss=0.53 | auc=0.631 | ok |
| knn-clf | istella | classical2 | 834 | 308 | sklearn-cpu | 292 | 2.85 | 1.05 |  | accuracy=0.92625 | accuracy=0.9263 | accuracy=0.9263 | ok |
| multioutput-reg | taxi | algos | 115 | 50.2 | sklearn-cpu | 53.7 | 2.14 | 0.94 | FLIP faster | r2=0.60424 | r2=0.6042 | r2=0.6043 | ok |
| iforest | istella | trees | 437 | 262 | sklearn-iforest-cpu (fill) | 304 | 1.44 | 0.86 | FLIP faster | auc=0.830358 | auc=0.8304 | auc=0.8279 | ok |
| gbdt-depthwise | istella | trees | 19062 | 17087 | xgboost-cpu (fill) | 23218 | 0.82 | 0.74 |  | logloss=0.156445, auc=0.983217 | auc=0.9802, logloss=0.1819 | auc=0.9836 | ok |
| dart-reg | istella | algos | 69437 | 23099 | lightgbm-cpu | 32389 | 2.14 | 0.71 | FLIP faster | r2=0.551355, rmse=0.559515 | r2=0.5507, rmse=0.5599 | r2=0.5647, rmse=0.5512 | ok |
| ocsvm | istella | algos | 446 | 451 | sklearn-cpu | 691 | 0.65 | 0.65 |  | fraction_flagged=0.0783 | fraction_flagged=0.0783 | fraction_flagged=0.0783, jaccard_vs_sklearn=1 | ok |
| dart | istella | algos | 69424 | 23014 | lightgbm-cpu | 35425 | 1.96 | 0.65 | FLIP faster | accuracy=0.9486, logloss=0.134461 | accuracy=0.9487, logloss=0.1341 | accuracy=0.9519, logloss=0.1237 | ok |
| knn | taxi | classical | 224 | 257 | sklearn-cpu | 426 | 0.53 | 0.60 |  | recall_at_k=0.999754, rows_with_repeated_ids=0 | recall_at_k=0.9998, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| gbdt-ordered | taxi | trees | - | 56105 | catboost-cpu (fill) | 99253 | - | 0.57 |  | logloss=0.52928, auc=0.628294 | - | auc=0.6277 | ok |
| gbdt-categorical | taxi | trees | 63398 | 32925 | lightgbm-cpu (fill) | 58904 | 1.08 | 0.56 | FLIP faster | logloss=0.528426, auc=0.63103 | auc=0.6302, logloss=0.5286 | auc=0.6327 | ok |
| nearest-centroid | istella | algos | 2896 | 263 | sklearn-cpu | 529 | 5.47 | 0.50 | FLIP faster | accuracy=0.85261, logloss=4.29948 | accuracy=0.8526, logloss=4.299 | accuracy=0.8526, logloss=4.118 | ok |
| enet-cv | taxi | algos | 3173 | 105 | sklearn-cpu | 211 | 15.03 | 0.50 | FLIP faster | r2=0.909004, rmse=4.80449 | r2=0.909, rmse=4.804 | r2=0.909, rmse=4.804 | ok |
| bayesian-ridge | istella | algos | 92368 | 3163 | sklearn-cpu | 6831 | 13.52 | 0.46 | FLIP faster | r2=nan, rmse=nan | r2=-4.169e+04, rmse=170.6 | r2=-890.2, rmse=24.94 | ok |
| lasso-cv | taxi | algos | 3170 | 104 | sklearn-cpu | 226 | 14.00 | 0.46 | FLIP faster | r2=0.909038, rmse=4.80359 | r2=0.9091, rmse=4.803 | r2=0.909, rmse=4.804 | ok |
| knn-clf | taxi | classical2 | 48.5 | 65.3 | sklearn-cpu | 154 | 0.32 | 0.42 |  | accuracy=0.74175 | accuracy=0.7418 | accuracy=0.7418 | ok |
| gbdt-symmetric-1000 | taxi | trees | 19086 | 22353 | catboost-cpu (fill) | 55261 | 0.35 | 0.40 |  | logloss=0.528216, auc=0.631675 | auc=0.6239, logloss=0.5306 | auc=0.6316 | ok |
| gbdt-symmetric | taxi | trees | 9520 | 11253 | catboost-cpu (fill) | 28188 | 0.34 | 0.40 |  | logloss=0.528595, auc=0.630376 | auc=0.6239, logloss=0.5306 | auc=0.6303 | ok |
| gbdt-lossguide | istella | trees | 106296 | 20547 | lightgbm-cpu (fill) | 55012 | 1.93 | 0.37 | FLIP faster | logloss=0.149694, auc=0.983676 | auc=0.9838, logloss=0.1488 | auc=0.9838 | ok |
| gbdt-rank-pairlogit | istella | trees | 3997 | 3064 | xgboost-cpu (fill) | 8355 | 0.48 | 0.37 |  | ndcg10=0.719953, ndcg5=0.6504, map=0.854545 | map=0.8414, ndcg10=0.7093, ndcg5=0.6398 | map=0.8728 | ok |
| gbdt-multiclass | taxi | trees | 14685 | 14938 | xgboost-cpu (fill) | 45500 | 0.32 | 0.33 |  | mlogloss=1.01259, accuracy=0.59938 | accuracy=0.5967, mlogloss=1.023 | accuracy=0.6011 | ok |
| gbdt-lossguide | taxi | trees | 44556 | 15377 | lightgbm-cpu (fill) | 51864 | 0.86 | 0.30 |  | logloss=0.528039, auc=0.631997 | auc=0.631, logloss=0.5283 | auc=0.6322 | ok |
| gbdt-symmetric | istella | trees | 11943 | 16955 | catboost-cpu (fill) | 60136 | 0.20 | 0.28 |  | logloss=0.186714, auc=0.980109 | auc=0.9756, logloss=0.2112 | auc=0.9799 | ok |
| gbdt-symmetric-1000 | istella | trees | 21364 | 32889 | catboost-cpu (fill) | 120805 | 0.18 | 0.27 |  | logloss=0.1704, auc=0.982434 | auc=0.9756, logloss=0.2112 | auc=0.9823 | ok |
| gbdt-multiclass | istella | trees | 21754 | 25728 | xgboost-cpu | 100166 | 0.22 | 0.26 |  | mlogloss=0.258413, accuracy=0.907556 | accuracy=0.9033, mlogloss=0.2809 | accuracy=0.9101, mlogloss=0.2468 | ok |
| rf | istella | trees | 13810 | 14232 | lightgbm-cpu | 67210 | 0.21 | 0.21 |  | logloss=0.182017, auc=0.945385 | auc=0.9454, logloss=0.182 | auc=0.9454, logloss=0.1954 | ok |
| rf | taxi | trees | 10796 | 11364 | lightgbm-cpu | 53864 | 0.20 | 0.21 |  | logloss=0.525953, auc=0.617838 | auc=0.6178, logloss=0.526 | auc=0.617, logloss=0.5264 | ok |
| dart | taxi | algos | 10520 | 5378 | lightgbm-cpu | 26498 | 0.40 | 0.20 |  | accuracy=0.76815, logloss=0.529149 | accuracy=0.7683, logloss=0.5291 | accuracy=0.7682, logloss=0.5291 | ok |
| dart-reg | taxi | algos | 9932 | 5199 | lightgbm-cpu | 27650 | 0.36 | 0.19 |  | r2=0.925497, rmse=4.34734 | r2=0.9255, rmse=4.347 | r2=0.9263, rmse=4.325 | ok |
| iforest | taxi | trees | 119 | 85.8 | sklearn-iforest-cpu (fill) | 575 | 0.21 | 0.15 |  | auc=0.551846 | auc=0.5518 | auc=0.5528 | ok |
| et | taxi | trees | 3376 | 3020 | sklearn-et-cpu | 20629 | 0.16 | 0.15 |  | logloss=0.526142, auc=0.618907 | auc=0.6189, logloss=0.5261 | auc=0.619, logloss=0.526 | ok |
| et | istella | trees | 4360 | 4305 | sklearn-et-cpu | 30617 | 0.14 | 0.14 |  | logloss=0.189989, auc=0.937987 | auc=0.938, logloss=0.19 | auc=0.9379, logloss=0.1901 | ok |
| multioutput-reg | istella | algos | 1823 | 1419 | sklearn-cpu | 13402 | 0.14 | 0.11 |  | r2=0.455327 | r2=0.4399 | r2=0.4553 | ok |
| lasso-cv | istella | algos | 70206 | 556 | sklearn-cpu | 5483 | 12.80 | 0.10 | FLIP faster | r2=0.325504, rmse=0.686041 | r2=0.3103, rmse=0.6937 | r2=0.3108, rmse=0.6935 | ok |
| damped-ets | taxi-hourly | algos | 612 | 17.4 | statsforecast-cpu | 185 | 3.31 | 0.09 | FLIP faster | forecast_rmse=96.6846 | forecast_rmse=96.69 | forecast_rmse=96.69 | ok |
| calibrated | taxi | algos | 1330 | 50.8 | sklearn-cpu | 556 | 2.39 | 0.09 | FLIP faster | accuracy=0.75533, logloss=0.550718 | accuracy=0.7553, logloss=0.5507 | accuracy=0.7553, logloss=0.5507 | ok |
| damped-ets | synthetic | algos | 530 | 15.2 | statsforecast-cpu | 167 | 3.17 | 0.09 | FLIP faster | forecast_rmse=13.9307 | forecast_rmse=13.93 | forecast_rmse=13.95 | ok |
| enet-cv | istella | algos | 74148 | 581 | sklearn-cpu | 6979 | 10.62 | 0.08 | FLIP faster | r2=0.326794, rmse=0.685384 | r2=0.3166, rmse=0.6906 | r2=0.3173, rmse=0.6902 | ok |
| ard | istella | algos | 47943 | 863 | sklearn-cpu | 10421 | 4.60 | 0.08 | FLIP faster | r2=-0.138726, rmse=0.891395 | r2=-0.1235, rmse=0.8854 | r2=0.3274, rmse=0.6851 | ok |
| calibrated | istella | algos | 2291 | 194 | sklearn-cpu | 2784 | 0.82 | 0.07 |  | accuracy=0.88509, logloss=0.289824 | accuracy=0.8851, logloss=0.2898 | accuracy=0.8851, logloss=0.2898 | ok |
| gbdt-categorical | taxicat | trees | - | 37105 | - | - | - | - |  | logloss=0.528399, auc=0.630905 | - | - | ok |
| gbdt-ordered | istella | trees | - | 75359 | - | - | - | - |  | logloss=0.190603, auc=0.979518 | - | - | ok |
| gbdt-rank-yetirank | istella | trees | 25773 | - | lightgbm-cpu (fill) | 7083 | 3.64 | - |  | - | map=0.8149, ndcg10=0.681, ndcg5=0.6151 | map=0.8584 | refused |

Sources: before = M3 0.8.34 board (classical), M3 2026-09-29 board FAST cells (trees); job tags afb2-algos-2, afb2-classical-4, afb2-classical2-3, afb2-trees-1b, afb3-cat-taxicat, afb3-trees-fast, afb4-grp-ocsvm-main, afb5-merged-algos, afb5-merged-trees, afb5b-dart.

## Quality flags (M3 manager, 2026-10-02, still open on 2026-10-03)

- **bayesian-ridge istella: NaN on main's FAST grid path.** The Gram sse shortcut cancels in f32 on istella; the row-pass
  arm (`MOJOLEARN_X_LINEAR_GRAM_SSE=0`) is finite (job br-main-q). Not counted as a win. Fix under test:
  `lane/apple-fast-bayes` `-D MOJOLEARN_BAYES_GRID_GUARD=1` (job bayes-br-guard-istella, next on the M3).
- **ard istella: r2 -0.139 vs sklearn 0.327.** FAST was already below sklearn before this pass (-0.124); the row-pass arm
  gives -0.122. Speed is real, quality is not at the opponent's level: open.
- **gbdt-rank-yetirank:** the refresh asked for dataset istella; the lane races on istellarank, so the driver refused.
  YetiRank's M3 FAST time on istellarank is 5.6 s (LEDGER), vs LightGBM 6.8 s on the board.
- Slower than before within this refresh: knn istella 1592 -> 1805 ms, ocsvm taxi 246 -> 372 ms, rf +3-5%. Before cells come
  from older boards (Sept 29 trees, 0.8.34 classical); not yet re-checked.
- OLS rows stay off this board: afb5-merged-classical OLS is a known regression under repair.
