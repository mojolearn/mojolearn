# M3 FAST board refresh (lane/apple-fast)

Our FAST arm on the M3 Ultra Metal GPU at head 0e743cac9, 1 warm-up + 3 timed rounds at board size (rows-full; trees MOJOLEARN_SPEED_SIZE=shipped). Opponents are not re-raced: classical times come from the M3 0.8.34 board (`~/mojolearn-evidence/board-0834-times.tsv`, quality from its board.json), trees from the 2026-09-29 M3 board (older tree params on some lanes). Ratio = our FAST ms / best opponent ms; below 1 is faster. Rows sort worst ratio after first. Written by `tools/af_board_merge.py`.

Summary: 25 rows, 24 with a ratio, 17 faster than the best opponent after (bayesian-ridge istella excluded: NaN), geometric-mean ratio 0.47. Flips to faster: iforest istella, nearest-centroid istella, enet-cv taxi, bayesian-ridge istella, lasso-cv taxi, gbdt-lossguide istella, gbdt-lossguide taxi, lasso-cv istella, enet-cv istella, ard istella. Flips to slower: none.

| lane | dataset | family | FAST before ms | FAST after ms | best opponent | opp ms | ratio before | ratio after | flip | quality after (FAST) | quality before (FAST) | opponent quality | status |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| bayesian-ridge | taxi | algos | 440 | 404 | sklearn-cpu | 73.2 | 6.01 | 5.52 |  | r2=0.908981, rmse=4.80511 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| knn | istella | classical | 1592 | 1805 | sklearn-cpu | 566 | 2.82 | 3.19 |  | recall_at_k=0.976613, rows_with_repeated_ids=0 | recall_at_k=0.9766, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| ard | taxi | algos | 44.3 | 41.9 | sklearn-cpu | 17.5 | 2.53 | 2.40 |  | r2=0.909193, rmse=4.79951 | r2=0.9092, rmse=4.8 | r2=0.9092, rmse=4.8 | ok |
| gbdt-depthwise | taxi | trees | 13994 | 15056 | xgboost-cpu | 9923 | 1.41 | 1.52 |  | logloss=0.527966, auc=0.63234 | auc=0.6258, logloss=0.53 | auc=0.631, logloss=0.5287 | ok |
| nearest-centroid | taxi | algos | 471 | 129 | sklearn-cpu | 96.9 | 4.86 | 1.33 |  | accuracy=0.6667, logloss=0.781905 | accuracy=0.6667, logloss=0.7822 | accuracy=0.6667, logloss=0.7817 | ok |
| knn-clf | istella | classical2 | 834 | 308 | sklearn-cpu | 292 | 2.85 | 1.05 |  | accuracy=0.92625 | accuracy=0.9263 | accuracy=0.9263 | ok |
| iforest | istella | trees | 437 | 262 | sklearn-iforest-cpu | 282 | 1.55 | 0.93 | FLIP faster | auc=0.830358 | auc=0.8304 | auc=0.8279 | ok |
| gbdt-depthwise | istella | trees | 19062 | 20005 | xgboost-cpu | 21590 | 0.88 | 0.93 |  | logloss=0.156598, auc=0.983275 | auc=0.9802, logloss=0.1819 | auc=0.9836, logloss=0.1493 | ok |
| knn | taxi | classical | 224 | 257 | sklearn-cpu | 426 | 0.53 | 0.60 |  | recall_at_k=0.999754, rows_with_repeated_ids=0 | recall_at_k=0.9998, rows_with_repeated_ids=0 | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| nearest-centroid | istella | algos | 2896 | 263 | sklearn-cpu | 529 | 5.47 | 0.50 | FLIP faster | accuracy=0.85261, logloss=4.29948 | accuracy=0.8526, logloss=4.299 | accuracy=0.8526, logloss=4.118 | ok |
| enet-cv | taxi | algos | 3173 | 105 | sklearn-cpu | 211 | 15.03 | 0.50 | FLIP faster | r2=0.909004, rmse=4.80449 | r2=0.909, rmse=4.804 | r2=0.909, rmse=4.804 | ok |
| bayesian-ridge | istella | algos | 92368 | 3163 | sklearn-cpu | 6831 | 13.52 | 0.46 | NOT A WIN: NaN | r2=nan, rmse=nan | r2=-4.169e+04, rmse=170.6 | r2=-890.2, rmse=24.94 | QUALITY FAIL |
| lasso-cv | taxi | algos | 3170 | 104 | sklearn-cpu | 226 | 14.00 | 0.46 | FLIP faster | r2=0.909038, rmse=4.80359 | r2=0.9091, rmse=4.803 | r2=0.909, rmse=4.804 | ok |
| knn-clf | taxi | classical2 | 48.5 | 65.3 | sklearn-cpu | 154 | 0.32 | 0.42 |  | accuracy=0.74175 | accuracy=0.7418 | accuracy=0.7418 | ok |
| gbdt-lossguide | istella | trees | 106296 | 20547 | lightgbm-cpu | 54183 | 1.96 | 0.38 | FLIP faster | logloss=0.149694, auc=0.983676 | auc=0.9838, logloss=0.1488 | auc=0.9838, logloss=0.1497 | ok |
| gbdt-lossguide | taxi | trees | 44556 | 15377 | lightgbm-cpu | 41350 | 1.08 | 0.37 | FLIP faster | logloss=0.528039, auc=0.631997 | auc=0.631, logloss=0.5283 | auc=0.6322, logloss=0.5281 | ok |
| rf | istella | trees | 13810 | 14232 | lightgbm-cpu | 67210 | 0.21 | 0.21 |  | logloss=0.182017, auc=0.945385 | auc=0.9454, logloss=0.182 | auc=0.9454, logloss=0.1954 | ok |
| rf | taxi | trees | 10796 | 11364 | lightgbm-cpu | 53864 | 0.20 | 0.21 |  | logloss=0.525953, auc=0.617838 | auc=0.6178, logloss=0.526 | auc=0.617, logloss=0.5264 | ok |
| iforest | taxi | trees | 119 | 85.8 | sklearn-iforest-cpu | 465 | 0.26 | 0.18 |  | auc=0.551846 | auc=0.5518 | auc=0.5528 | ok |
| et | taxi | trees | 3376 | 3020 | sklearn-et-cpu | 20629 | 0.16 | 0.15 |  | logloss=0.526142, auc=0.618907 | auc=0.6189, logloss=0.5261 | auc=0.619, logloss=0.526 | ok |
| et | istella | trees | 4360 | 4305 | sklearn-et-cpu | 30617 | 0.14 | 0.14 |  | logloss=0.189989, auc=0.937987 | auc=0.938, logloss=0.19 | auc=0.9379, logloss=0.1901 | ok |
| lasso-cv | istella | algos | 70206 | 556 | sklearn-cpu | 5483 | 12.80 | 0.10 | FLIP faster | r2=0.325504, rmse=0.686041 | r2=0.3103, rmse=0.6937 | r2=0.3108, rmse=0.6935 | ok |
| enet-cv | istella | algos | 74148 | 581 | sklearn-cpu | 6979 | 10.62 | 0.08 | FLIP faster | r2=0.326794, rmse=0.685384 | r2=0.3166, rmse=0.6906 | r2=0.3173, rmse=0.6902 | ok |
| ard | istella | algos | 47943 | 863 | sklearn-cpu | 10421 | 4.60 | 0.08 | FLIP faster | r2=-0.138726, rmse=0.891395 | r2=-0.1235, rmse=0.8854 | r2=0.3274, rmse=0.6851 | QUALITY BELOW OPPONENT |
| gbdt-rank-yetirank | istella | trees | 25773 | - | lightgbm-cpu | 6765 | 3.81 | - |  | - | map=0.8149, ndcg10=0.681, ndcg5=0.6151 | map=0.8584, ndcg10=0.7415, ndcg5=0.6803 | refused |

Sources: before = M3 0.8.34 board (classical), M3 2026-09-29 board ours-ab FAST cells (trees); job tags afb2-algos-2, afb2-classical-4, afb2-classical2-3, afb2-trees-1b.

## Quality flags (M3 manager, 2026-10-02)

- **bayesian-ridge istella: NaN on main's FAST grid path.** The Gram sse shortcut cancels in f32 on istella; the row-pass
  arm (`MOJOLEARN_X_LINEAR_GRAM_SSE=0`) is finite (job br-main-q). Not counted as a win. Fix under test:
  `lane/apple-fast-bayes` `-D MOJOLEARN_BAYES_GRID_GUARD=1` (job bayes-br-guard-istella, next on the M3).
- **ard istella: r2 -0.139 vs sklearn 0.327.** FAST was already below sklearn before this pass (-0.124); the row-pass arm
  gives -0.122. Speed is real, quality is not at the opponent's level: open.
- **gbdt-rank-yetirank:** the refresh asked for dataset istella; the lane races on istellarank, so the driver refused.
  YetiRank's M3 FAST time on istellarank is 5.6 s (LEDGER), vs LightGBM 6.8 s on the board.
- Slower than before within this refresh: knn istella 1592 -> 1805 ms, gbdt-depthwise taxi 13.99 -> 15.06 s / istella
  19.06 -> 20.00 s, rf +3-5%. Before cells come from older boards (Sept 29 trees, 0.8.34 classical); not yet re-checked.
