# M3 IDENTICAL board

Generated from board.json by tools/af_board_render.py; do not edit. Source: `bench/results/bench_board/m3ultra-0834/board.json` (the one M3 board; BOARD.md, BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md are rendered from it).

## Checks

- IDENTICAL cells come from M3 ident sweeps (`tools/af_board_ident_update.py`, which writes board.json and re-renders every page); same-bits checks are separate (`lq add <box> ID ...`).
- Render check: `python3 tools/af_board_render.py --check` fails when BOARD.md, BOARD_M3_FAST.md or BOARD_M3_IDENTICAL.md differ from a fresh render of board.json. `tools/af_board_apply.py` and `tools/af_board_ident_update.py` always run it; `tools/af_board_render_watch.sh` (called from ~/mojolearn-evidence/apple_watch.sh) runs it against origin/main and prints ALERT on a mismatch.

Our IDENTICAL arm on the M3 Ultra Metal GPU at head cf94a6be6, 1 unscored warm-up + 1 scored round at board size (the 2026-10-04 ident-* sweep, 350 jobs, AFC_ARM=ours) (rows-full; trees MOJOLEARN_SPEED_SIZE=shipped). Opponents are not re-raced: classical times come from the M3 0.8.34 board (`~/mojolearn-evidence/board-0834-times.tsv`, quality from its board.json), trees from the 2026-09-29 M3 board (older tree params on some lanes); an opponent marked (fill) comes from the M3 opponent fill on the 0.8.34 kit. Ratio = our IDENTICAL ms / best opponent ms; below 1 is faster. Rows sort worst ratio after first. First written by `tools/af_board_merge.py`; now carried in board.json.

Quality (2026-10-04, computed): of 429 lane/dataset rows, IDENTICAL quality is WORSE than an opponent on 69 (49 by more than 1%) and IDENTICAL after is WORSE than IDENTICAL before on 0 (0 by more than 1%); 2 rows carry metrics of unknown direction (1 metrics), 81 have no parseable IDENTICAL quality. Tolerance rel 1e-3, abs 1e-6.

Canonical full-board summary (2026-10-04): 421 IDENTICAL rows, 413 eligible opponent comparisons, 237 faster, geometric-mean ratio 0.612. Eligible = a ratio after and a status without HOLD or "excluded"; faster = ratio below 1. Ratio = IDENTICAL ms / best opponent ms.

- Hand-made headline, last before the page was generated: Summary: 350 rows, 329 with a ratio, 203 faster than the best opponent after, geometric-mean ratio 0.48. Flips to faster: ard taxi, svd taxi, select-r-regression istella, gpr taxi, svd istella, enet-cv taxi, bisecting-kmeans taxi, connected-components istella, label-encoder istella, perceptron istella, multinomial-nb istella, complement-nb istella, lasso-cv taxi, minmax-scaler taxi, maxabs-scaler taxi, robust-scaler taxi, tree-shap istella, nearest-centroid istella, meanshift istella, isotonic taxi, pa-clf istella, ridge-clf istella, dict-learning istella, mb-sparse-pca istella, complement-nb taxi, multinomial-nb taxi, label-encoder taxi, gaussian-nb istella, lasso-cv istella, qr taxi, pa-reg istella, tree-shap taxi, enet-cv istella, lstsq istella, gaussian-nb taxi, categorical-nb taxi, categorical-nb istella, bernoulli-nb taxi, lda taxi-zones, huber taxi, bayesian-ridge istella, ard istella, huber istella. Flips to slower: pagerank taxi, pagerank istella, umap istella, rbf-sampler taxi, ridge taxi, permutation-shap taxi, kernel-shap taxi, louvain istella, spectral taxi, ols taxi, ivf istella, select-f-regression istella.

| lane | dataset | family | IDENTICAL before ms | IDENTICAL after ms | best opponent | opp ms | ratio before | ratio after | flip | quality after (IDENTICAL) | quality before (IDENTICAL) | opponent quality | status |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| cagra | istella | algos | - | 181960 | faiss-cpu | 1634 | - | 111.39 |  | recall_at_10=0.9838 | - | recall_at_10=0.9994 | ok |
| sgd-ocsvm | taxi | algos | 4333 | 91217 | sklearn-cpu | 898 | 4.82 | 101.57 |  | fraction_flagged=0.28379 | - | fraction_flagged=0.00702, jaccard_vs_sklearn=1 | ok |
| cross-entropy | synthetic | algos | 324 | 324 | torch-compile-fp32 | 3.6 | 89.33 | 89.33 |  | loss_rel_err_vs_fp64=4.564e-08 | loss_rel_err_vs_fp64=4.564e-08 | grad_max_rel_diff_vs_ours=5.588e-09, loss_rel_err_vs_fp64=4.564e-08 | ok |
| avgpool2d | synthetic | algos | 146 | 146 | torch-compile-bf16 | 2.5 | 57.32 | 57.32 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| min-cov-det | taxi | algos | 3919 | 80018 | sklearn-cpu | 1413 | 2.77 | 56.61 |  | n_features=11 | - | n_features=11 | ok |
| elliptic-envelope | taxi | algos | 3948 | 79473 | sklearn-cpu | 1423 | 2.77 | 55.84 |  | fraction_flagged=0.10237 | - | fraction_flagged=0.1025, jaccard_vs_sklearn=1 | ok |
| resnet-block | synthetic | algos | 486 | 486 | torch-compile-bf16 | 10.5 | 46.44 | 46.44 |  | - | - | max_rel_diff_vs_torch_eager_fp32=1.85e+04, rel_fro_vs_torch_eager_fp32=0.003425 | ok |
| graphsage | istella | algos | 387 | 387 | torch-compile-bf16 | 10.2 | 38.02 | 38.02 |  | - | - | max_rel_diff_vs_torch_eager_fp32=3678, rel_fro_vs_torch_eager_fp32=0.003054 | ok |
| maxpool2d | synthetic | algos | 156 | 156 | torch-eager-bf16 | 4.2 | 37.34 | 37.34 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| dropout2d | synthetic | algos | 82.7 | 82.7 | torch-compile-fp32 | 2.2 | 37.05 | 37.05 |  | - | - | - | ok |
| maxpool1d | synthetic | algos | 63.1 | 63.1 | torch-compile-bf16 | 2.0 | 30.93 | 30.93 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| batchnorm1d | synthetic | algos | 72.6 | 72.6 | torch-eager-bf16 | 2.4 | 30.42 | 30.42 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| conv1d | synthetic | algos | 237 | 237 | torch-compile-bf16 | 8.0 | 29.67 | 29.67 |  | - | - | max_rel_diff_vs_torch_eager_fp32=3418, rel_fro_vs_torch_eager_fp32=0.003296 | ok |
| avgpool1d | synthetic | algos | 50.0 | 50.0 | torch-compile-fp32 | 1.7 | 29.40 | 29.40 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| batchnorm2d | synthetic | algos | 62.1 | 62.1 | torch-eager-bf16 | 2.3 | 26.71 | 26.71 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| moe | synthetic | algos | 820 | 820 | torch-eager-bf16 | 37.7 | 21.74 | 21.74 |  | - | - | max_rel_diff_vs_torch_eager_fp32=2.234e+04, rel_fro_vs_torch_eager_fp32=0.05522 | ok |
| autoarima | taxi-hourly | algos | - | 57797 | statsforecast-cpu | 2844 | - | 20.32 |  | forecast_rmse=73.6303 | - | forecast_rmse=68.21 | ok |
| conv2d | synthetic | algos | 109 | 109 | torch-compile-bf16 | 5.4 | 20.18 | 20.18 |  | - | - | max_rel_diff_vs_torch_eager_fp32=3418, rel_fro_vs_torch_eager_fp32=0.003383 | ok |
| graphsage | taxi | algos | 123 | 123 | torch-compile-bf16 | 6.2 | 20.04 | 20.04 |  | - | - | max_rel_diff_vs_torch_eager_fp32=4608, rel_fro_vs_torch_eager_fp32=0.003285 | ok |
| sgd-ocsvm | istella | algos | 69730 | 107029 | sklearn-cpu | 5734 | 12.16 | 18.67 |  | fraction_flagged=0.2176 | - | fraction_flagged=0.09334, jaccard_vs_sklearn=1 | ok |
| autoarima | synthetic | algos | - | 39660 | statsforecast-cpu | 2880 | - | 13.77 |  | forecast_rmse=3.41284 | - | forecast_rmse=17.55 | ok |
| lr-onecycle | synthetic | algos | 1385 | 1385 | torch-cpu | 111 | 12.50 | 12.50 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.951e-08 | ok |
| gcn | istella | algos | 194 | 194 | torch-compile-fp32 | 18.8 | 10.34 | 10.34 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0.01759, rel_fro_vs_torch_eager_fp32=1.004e-07 | ok |
| layernorm | synthetic | algos | 27.6 | 27.6 | torch-eager-bf16 | 2.8 | 10.02 | 10.02 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| spline | istella | algos | 166 | 166 | sklearn-cpu | 17.6 | 9.44 | 9.45 |  | - | - | - | ok |
| adagrad | synthetic | algos | 131 | 131 | torch-eager-fp32 | 14.2 | 9.22 | 9.22 |  | - | - | - | ok |
| spline | taxi | algos | 148 | 147 | sklearn-cpu | 17.6 | 8.38 | 8.31 |  | - | - | - | ok |
| gcn | taxi | algos | 144 | 144 | torch-compile-fp32 | 17.8 | 8.13 | 8.13 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0.003725, rel_fro_vs_torch_eager_fp32=9.327e-08 | ok |
| lm-forward | bytes | neural | 93.3 | 93.3 | torch-compile-bf16 | 12.0 | 7.76 | 7.76 |  | mean_nll=9.019 | mean_nll=9.019 | max_abs_diff_vs_ours=0.00671, max_rel_diff_vs_ours=0.009211, mean_nll=9.019 | ok |
| sparse-coder | istella | algos | 0.2 | 0.2 | sklearn-cpu | 0.0 | 7.59 | 7.70 |  | - | - | - | ok |
| lr-warmup-linear | synthetic | algos | 780 | 780 | torch-cpu | 101 | 7.68 | 7.68 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0.001 | ok |
| lr-constant | synthetic | algos | 570 | 570 | torch-cpu | 87.4 | 6.52 | 6.52 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=1.038e-07 | ok |
| adamax | synthetic | algos | 176 | 176 | torch-eager-fp32 | 27.7 | 6.35 | 6.35 |  | - | - | - | ok |
| gemm | gaussian | neural | 72.3 | 72.3 | torch-compile-bf16 | 11.5 | 6.30 | 6.30 |  | max_rel_err_vs_fp64=2.399e-07 | max_rel_err_vs_fp64=2.399e-07 | max_abs_diff_vs_ours=1.358, max_rel_diff_vs_ours=0.003752, max_rel_err_vs_fp64=0.003752 | ok |
| garch | taxi-hourly | algos | 133 | 772 | arch-cpu | 125 | 1.07 | 6.19 |  | mean_llf=-1132.71 | - | mean_llf=-1130 | ok |
| damped-ets | synthetic | algos | 1017 | 1017 | statsforecast-cpu | 167 | 6.08 | 6.08 |  | forecast_rmse=13.9312 | - | forecast_rmse=13.95 | ok |
| damped-ets | taxi-hourly | algos | 1124 | 1121 | statsforecast-cpu | 185 | 6.07 | 6.06 |  | forecast_rmse=96.6904 | - | forecast_rmse=96.69 | ok |
| garch | synthetic | algos | 131 | 626 | arch-cpu | 106 | 1.25 | 5.93 |  | mean_llf=-1938.22 | - | mean_llf=-1938 | ok |
| lda-clf | istella | algos | 21856 | 22056 | sklearn-cpu | 3739 | 5.85 | 5.90 |  | accuracy=0.91166, logloss=0.247483 | - | accuracy=0.9011, logloss=0.4492 | ok |
| nadam | synthetic | algos | 180 | 180 | torch-eager-fp32 | 32.9 | 5.46 | 5.46 |  | - | - | - | ok |
| select-f-classif | taxi | algos | 215 | 216 | sklearn-cpu | 41.1 | 5.24 | 5.26 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| rmsprop | synthetic | algos | 131 | 131 | torch-eager-fp32 | 25.2 | 5.21 | 5.21 |  | - | - | - | ok |
| label-binarizer | taxi | algos | 1360 | 559 | sklearn-cpu | 118 | 11.57 | 4.75 |  | - | - | - | ok |
| sparse-coder | taxi | algos | 0.2 | 0.2 | sklearn-cpu | 0.0 | 3.64 | 4.74 |  | - | - | - | ok |
| lr-warmup-cosine | synthetic | algos | 539 | 539 | torch-cpu | 114 | 4.74 | 4.74 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | - | ok |
| onehot | taxi | algos | 84.6 | 89.5 | sklearn-cpu | 19.0 | 4.45 | 4.71 |  | - | - | - | ok |
| select-f-regression | taxi | algos | 145 | 146 | sklearn-cpu | 31.3 | 4.64 | 4.66 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| gemm-bf16 | gaussian | neural | 69.1 | 69.1 | torch-eager-bf16 | 14.8 | 4.65 | 4.65 |  | max_rel_err_vs_fp64=1.155e-07 | max_rel_err_vs_fp64=1.155e-07 | max_abs_diff_vs_ours=0.999, max_rel_diff_vs_ours=0.002759, max_rel_err_vs_fp64=0.002759 | ok |
| adafactor | synthetic | algos | 1588 | 1588 | torch-eager-fp32 | 343 | 4.62 | 4.62 |  | - | - | - | ok |
| select-r-regression | taxi | algos | 147 | 146 | sklearn-cpu | 31.8 | 4.63 | 4.59 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| ridge-cv | taxi | algos | - | 4306 | sklearn-cpu | 946 | - | 4.55 |  | finite=1, r2=0.908981, rmse=4.80511 | - | r2=0.909, rmse=4.805 | ok |
| ordinal | taxi | algos | 83.1 | 83.9 | sklearn-cpu | 18.8 | 4.43 | 4.47 |  | - | - | - | ok |
| lm-train-step | bytes | neural | 185 | 185 | torch-compile-bf16 | 42.9 | 4.31 | 4.31 |  | loss_first_step=9.019, loss_last_step=8.418, steps=2 | loss_first_step=9.019, loss_last_step=8.418, steps=2 | loss_first_abs_diff_vs_ours=0.0002089, loss_first_step=9.018, loss_last_abs_diff_vs_ours=0 | ok |
| svgp | taxi | algos | - | 1040 | gpytorch-cpu | 248 | - | 4.20 |  | finite=1, r2=-0.194982, rmse=17.7235 | - | r2=-0.2093, rmse=17.83 | ok |
| cholesky | synthetic | algos | 994 | 445 | torch-gpu | 110 | 9.01 | 4.04 |  | relative_residual=2.90007e-07 | - | relative_residual=5.449e-07 | ok |
| pca | istella | classical | 814 | 819 | sklearn-cpu | 205 | 3.97 | 3.99 |  | explained_variance_ratio_sum=1 | - | explained_variance_ratio_sum=1 | ok |
| stl | synthetic | algos | 414 | 414 | statsmodels-cpu | 105 | 3.95 | 3.95 |  | residual_std=0.783175 | - | residual_std=0.7832 | ok |
| select-chi2 | taxi | algos | 208 | 209 | sklearn-cpu | 58.5 | 3.55 | 3.58 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| svgp | istella | algos | 3718 | 1094 | gpytorch-cpu | 307 | 12.10 | 3.56 |  | finite=1, r2=-0.106016, rmse=0.878373 | - | r2=-0.106, rmse=0.8784 | ok |
| kernel-shap | istella | algos | 14675 | 27059 | shap-cpu | 7695 | 1.91 | 3.52 |  | rel_error_vs_exact=4.37806e-09 | - | rel_error_vs_exact=1.405e-14 | ok |
| stl | taxi-hourly | algos | 414 | 415 | statsmodels-cpu | 119 | 3.49 | 3.49 |  | residual_std=18.3125 | - | residual_std=18.31 | ok |
| global-avgpool | synthetic | algos | 4.1 | 4.1 | torch-compile-bf16 | 1.2 | 3.48 | 3.48 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=9.321e-08 | ok |
| cagra | taxi | algos | - | 3532 | faiss-cpu | 1016 | - | 3.48 |  | recall_at_10=0.4838 | - | recall_at_10=0.9277 | ok |
| pagerank | taxi | algos | 45.2 | 226 | networkx-cpu | 64.9 | 0.70 | 3.48 | FLIP slower | sum=1 | - | sum=1 | ok |
| lu-factor | synthetic | algos | 2381 | 1429 | scipy-cpu | 428 | 5.56 | 3.34 |  | relative_residual=3.2563e-06 | - | relative_residual=3.246e-06 | ok |
| minibatch-kmeans | istella | algos | 690 | 405 | sklearn-cpu | 123 | 5.58 | 3.28 |  | n_clusters=8, silhouette=0.116696 | - | ari_vs_ours=0.6224, n_clusters=8, silhouette=0.1119 | ok |
| global-maxpool | synthetic | algos | 4.3 | 4.3 | torch-eager-bf16 | 1.3 | 3.27 | 3.27 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| samba-train-step | bytes | neural | 339 | 339 | torch-eager-bf16 | 106 | 3.19 | 3.19 |  | loss_first_step=5.636, loss_last_step=4.834, steps=2 | loss_first_step=5.636, loss_last_step=4.834, steps=2 | loss_first_abs_diff_vs_ours=9.632e-05, loss_first_step=5.636, loss_last_abs_diff_vs_ours=0 | ok |
| lle | taxi | algos | 18498 | 3951 | sklearn-cpu | 1247 | 14.84 | 3.17 |  | trustworthiness_k15=0.826083 | - | trustworthiness_k15=0.7708 | ok |
| skewed-chi2 | taxi | algos | 3.6 | 1.7 | sklearn-cpu | 0.6 | 6.49 | 3.07 |  | kernel_rel_error=0.0377486 | - | kernel_rel_error=0.03775 | ok |
| onehot | istella | algos | 110 | 111 | sklearn-cpu | 36.7 | 3.01 | 3.03 |  | - | - | - | ok |
| ordinal | istella | algos | 111 | 111 | sklearn-cpu | 36.9 | 3.02 | 3.01 |  | - | - | - | ok |
| transformer-forward | gaussian | neural | 14.1 | 14.1 | torch-compile-bf16 | 4.7 | 3.00 | 3.00 |  | - | - | max_abs_diff_vs_ours=0.001819, max_rel_diff_vs_ours=0.0003893 | ok |
| qda | istella | algos | 18931 | 19056 | sklearn-cpu | 6397 | 2.96 | 2.98 |  | accuracy=0.86609, logloss=4.05287 | - | accuracy=0.8805, logloss=3.477 | ok |
| lr-exponential | synthetic | algos | 225 | 225 | torch-cpu | 76.7 | 2.94 | 2.94 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.933e-08 | ok |
| multioutput-reg | taxi | algos | 105 | 155 | sklearn-cpu | 53.7 | 1.96 | 2.90 |  | r2=0.60424 | - | r2=0.6043 | ok |
| additive-chi2 | istella | algos | 11.0 | 11.0 | sklearn-cpu | 3.8 | 2.88 | 2.89 |  | kernel_rel_error=0.0877304 | - | kernel_rel_error=0.08773 | ok |
| lstm-reg | synthetic | algos | 1984 | 1984 | torch-compile-fp32 (fill) | 695 | 2.85 | 2.85 |  | r2=0.981, rmse=0.1596 | r2=0.981, rmse=0.1596 | - | ok |
| lstm-reg | taxi-hourly | algos | 1989 | 1989 | torch-compile-fp32 (fill) | 700 | 2.84 | 2.84 |  | r2=0.7517, rmse=0.5404 | r2=0.7517, rmse=0.5404 | - | ok |
| lu-solve | synthetic | algos | 2363 | 1433 | torch-gpu | 505 | 4.67 | 2.83 |  | relative_residual=3.2563e-06 | - | relative_residual=8.234e-07 | ok |
| permutation-shap | istella | algos | 16956 | 34179 | shap-cpu | 12335 | 1.37 | 2.77 |  | rel_error_vs_exact=5.33875e-09 | - | rel_error_vs_exact=3.692e-10 | ok |
| target-encoder | taxi | algos | 525 | 407 | sklearn-cpu | 147 | 3.56 | 2.76 |  | - | - | - | ok |
| pagerank | istella | algos | 47.0 | 224 | networkx-cpu | 82.0 | 0.57 | 2.73 | FLIP slower | sum=1 | - | sum=1 | ok |
| lstm-clf | taxi-hourly | algos | 1991 | 1991 | torch-eager-fp32 (fill) | 729 | 2.73 | 2.73 |  | accuracy=0.8682, logloss=0.2999 | accuracy=0.8682, logloss=0.2999 | - | ok |
| lstm-clf | synthetic | algos | 1991 | 1991 | torch-eager-fp32 (fill) | 729 | 2.73 | 2.73 |  | accuracy=0.9687, logloss=0.07244 | accuracy=0.9687, logloss=0.07244 | - | ok |
| nystroem | taxi | classical2 | 629 | 659 | sklearn-cpu | 246 | 2.56 | 2.68 |  | kernel_rel_error=0.0456109 | - | kernel_rel_error=0.04437 | ok |
| calibrated | taxi | algos | 2624 | 1422 | sklearn-cpu | 556 | 4.72 | 2.56 |  | accuracy=0.75533, logloss=0.550718 | - | accuracy=0.7553, logloss=0.5507 | ok |
| variance-threshold | taxi | algos | 127 | 125 | sklearn-cpu | 50.6 | 2.51 | 2.48 |  | - | - | - | ok |
| lars | taxi | algos | 208 | 82.1 | sklearn-cpu | 34.2 | 6.08 | 2.40 |  | finite=1, r2=0.908981, rmse=4.80511 | - | r2=0.909, rmse=4.805 | ok |
| lasso-lars | taxi | algos | 210 | 82.5 | sklearn-cpu | 34.6 | 6.07 | 2.38 |  | finite=1, r2=0.908996, rmse=4.8047 | - | r2=0.909, rmse=4.804 | ok |
| clip-grad-norm | synthetic | algos | 40.5 | 40.5 | torch-compile-fp32 | 17.0 | 2.37 | 2.37 |  | norm=1, norm_rel_diff_vs_ours=0 | norm=1, norm_rel_diff_vs_ours=0 | norm=1, norm_rel_diff_vs_ours=0 | ok |
| dynamic-optimized-theta | taxi-hourly | algos | 3178 | 3196 | statsforecast-cpu | 1352 | 2.35 | 2.36 |  | forecast_rmse=49.0866 | - | forecast_rmse=49.31 | ok |
| gaussian-rp | taxi | algos | 4.5 | 3.8 | sklearn-cpu | 1.6 | 2.80 | 2.36 |  | mean_abs_distortion=0.345752 | - | mean_abs_distortion=0.3398 | ok |
| minmax-scaler | istella | algos | 355 | 143 | sklearn-cpu | 63.0 | 5.63 | 2.27 |  | - | - | - | ok |
| theta | taxi-hourly | algos | 388 | 385 | statsmodels-cpu | 171 | 2.27 | 2.25 |  | forecast_rmse=49.0206 | - | forecast_rmse=49.31 | ok |
| lasso | istella | classical2 | 8496 | 8496 | sklearn-cpu | 3839 | 2.21 | 2.21 |  | r2=0.3108, rmse=0.6935 | r2=0.3108, rmse=0.6935 | r2=0.3108, rmse=0.6935 | ok |
| gaussian-rp | istella | algos | 57.3 | 53.8 | sklearn-cpu | 24.3 | 2.36 | 2.21 |  | mean_abs_distortion=0.680693 | - | mean_abs_distortion=0.178 | ok |
| sparse-rp | taxi | algos | 8.0 | 3.9 | sklearn-cpu | 1.8 | 4.48 | 2.19 |  | mean_abs_distortion=0.147163 | - | mean_abs_distortion=0.381 | ok |
| multilabel-binarizer | taxi | algos | 723 | 283 | sklearn-cpu | 129 | 5.59 | 2.19 |  | - | - | - | ok |
| sparse-rp | istella | algos | 66.8 | 54.5 | sklearn-cpu | 25.3 | 2.64 | 2.16 |  | mean_abs_distortion=1.88338 | - | mean_abs_distortion=0.4743 | ok |
| rbf-sampler | istella | classical2 | 62.7 | 101 | sklearn-cpu | 47.6 | 1.32 | 2.13 |  | kernel_rel_error=0.14198 | - | kernel_rel_error=0.1374 | ok |
| ocsvm | taxi | algos | 263 | 383 | sklearn-cpu | 181 | 1.45 | 2.12 |  | fraction_flagged=0.1361 | - | fraction_flagged=0.1361, jaccard_vs_sklearn=1 | ok |
| umap | istella | classical2 | 1454 | 3349 | umap-learn-cpu-unseeded | 1586 | 0.92 | 2.11 | FLIP slower | trustworthiness_k15=0.977793 | - | trustworthiness_k15=0.9766 | ok |
| mlp-train-step | gaussian | neural | 8.4 | 8.4 | torch-compile-fp32 | 4.0 | 2.09 | 2.09 |  | loss_first_step=1.16, loss_last_step=1.123, steps=2 | loss_first_step=1.16, loss_last_step=1.123, steps=2 | loss_first_abs_diff_vs_ours=2.384e-07, loss_first_step=1.16, loss_last_abs_diff_vs_ours=2. | ok |
| kernel-pca | taxi | algos | 877 | 949 | sklearn-cpu | 461 | 1.90 | 2.06 |  | - | - | subspace_cos_vs_sklearn=1 | ok |
| skewed-chi2 | istella | algos | 8.3 | 7.5 | sklearn-cpu | 3.7 | 2.21 | 1.99 |  | kernel_rel_error=0.671898 | - | kernel_rel_error=0.6719 | ok |
| var | synthetic | algos | 19.5 | 5.3 | statsmodels-cpu | 2.6 | 7.37 | 1.99 |  | forecast_rmse=1.14086 | - | forecast_rmse=1.145 | ok |
| nearest-centroid | taxi | algos | 564 | 192 | sklearn-cpu | 96.9 | 5.83 | 1.98 |  | accuracy=0.66675, logloss=0.782162 | - | accuracy=0.6667, logloss=0.7817 | ok |
| minibatch-kmeans | taxi | algos | 108 | 87.0 | sklearn-cpu | 43.9 | 2.45 | 1.98 |  | n_clusters=8, silhouette=0.13806 | - | ari_vs_ours=0.5252, n_clusters=8, silhouette=0.1655 | ok |
| randomized-svd | istella | algos | 725 | 718 | sklearn-cpu | 369 | 1.96 | 1.95 |  | relative_reconstruction_error=0.000235946 | - | relative_reconstruction_error=0.0002359 | ok |
| var | taxi-hourly | algos | 19.6 | 5.2 | statsmodels-cpu | 2.8 | 7.06 | 1.88 |  | forecast_rmse=33.168 | - | forecast_rmse=33.17 | ok |
| kernel-pca | istella | algos | 1047 | 1036 | sklearn-cpu | 560 | 1.87 | 1.85 |  | - | - | subspace_cos_vs_sklearn=1 | ok |
| calibrated | istella | algos | 3339 | 5126 | sklearn-cpu | 2784 | 1.20 | 1.84 |  | accuracy=0.88509, logloss=0.289832 | - | accuracy=0.8851, logloss=0.2898 | ok |
| target-encoder | istella | algos | 531 | 411 | sklearn-cpu | 229 | 2.32 | 1.80 |  | - | - | - | ok |
| bayesian-ridge | taxi | algos | 482 | 131 | sklearn-cpu | 73.2 | 6.58 | 1.79 |  | finite=1, r2=0.908979, rmse=4.80515 | - | r2=0.909, rmse=4.805 | ok |
| additive-chi2 | taxi | algos | 0.7 | 0.7 | sklearn-cpu | 0.4 | 1.69 | 1.79 |  | kernel_rel_error=0.0938923 | - | kernel_rel_error=0.09389 | ok |
| elasticnet | taxi | classical2 | 68.5 | 68.5 | sklearn-cpu | 40.6 | 1.69 | 1.69 |  | r2=0.9074, rmse=4.847 | r2=0.9074, rmse=4.847 | r2=0.9074, rmse=4.847 | ok |
| kbins | taxi | algos | 185 | 183 | sklearn-cpu | 108 | 1.71 | 1.68 |  | - | - | - | ok |
| iterative-imputer | taxi | algos | 1761 | 1760 | sklearn-cpu | 1049 | 1.68 | 1.68 |  | masked_rmse=4.69385 | - | masked_rmse=4.694 | ok |
| bisecting-kmeans | istella | algos | 4521 | 2351 | sklearn-cpu | 1428 | 3.17 | 1.65 |  | n_clusters=8, silhouette=0.118345 | - | ari_vs_ours=0.6929, n_clusters=8, silhouette=0.09641 | ok |
| knn | taxi | classical | 730 | 699 | sklearn-cpu | 426 | 1.71 | 1.64 |  | recall_at_k=0.999754, rows_with_repeated_ids=0 | - | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| elasticnet | istella | classical2 | 2949 | 2949 | sklearn-cpu | 1809 | 1.63 | 1.63 |  | r2=0.2609, rmse=0.7181 | r2=0.2609, rmse=0.7181 | r2=0.2609, rmse=0.7181 | ok |
| lasso | taxi | classical2 | 76.1 | 76.1 | sklearn-cpu | 47.9 | 1.59 | 1.59 |  | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| lda-clf | taxi | algos | 289 | 292 | sklearn-cpu | 186 | 1.56 | 1.58 |  | accuracy=0.76258, logloss=0.539749 | - | accuracy=0.7625, logloss=0.5398 | ok |
| rbf-sampler | taxi | classical2 | 28.2 | 63.7 | sklearn-cpu | 42.2 | 0.67 | 1.51 | FLIP slower | kernel_rel_error=0.108549 | - | kernel_rel_error=0.08378 | ok |
| ridge | taxi | classical2 | 27.0 | 53.3 | sklearn-cpu | 35.7 | 0.75 | 1.49 | FLIP slower | finite=1, r2=0.908983, rmse=4.80504 | - | r2=0.909, rmse=4.805 | ok |
| qda | taxi | algos | 240 | 210 | sklearn-cpu | 141 | 1.71 | 1.49 |  | accuracy=0.72702, logloss=1.061 | - | accuracy=0.7272, logloss=1.059 | ok |
| kpss | taxi-hourly | algos | 4.3 | 3.8 | statsmodels-cpu | 2.6 | 1.67 | 1.48 |  | stationary_fraction=0.6875 | - | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| isotonic | istella | algos | 610 | 55.2 | sklearn-cpu | 37.5 | 16.28 | 1.47 |  | finite=1, r2=0.187985, rmse=0.752735 | - | r2=0.188, rmse=0.7527 | ok |
| label-binarizer | istella | algos | 322 | 46.6 | sklearn-cpu | 31.7 | 10.17 | 1.47 |  | - | - | - | ok |
| permutation-shap | taxi | algos | 68.1 | 122 | shap-cpu | 83.0 | 0.82 | 1.46 | FLIP slower | rel_error_vs_exact=2.1497e-08 | - | rel_error_vs_exact=1.279e-15 | ok |
| ivf-pq | istella | algos | 10338 | 10146 | faiss-cpu | 7008 | 1.48 | 1.45 |  | recall_at_10=0.550825 | - | recall_at_10=0.803 | ok |
| select-d | taxi-hourly | algos | 5.6 | 4.8 | statsmodels-cpu | 3.4 | 1.63 | 1.42 |  | - | - | d_agreement_vs_statsmodels=1 | ok |
| lle | istella | algos | 18395 | 4044 | sklearn-cpu | 2852 | 6.45 | 1.42 |  | trustworthiness_k15=0.895277 | - | trustworthiness_k15=0.8491 | ok |
| kpss | synthetic | algos | 3.6 | 3.7 | statsmodels-cpu | 2.6 | 1.37 | 1.41 |  | stationary_fraction=0.03125 | - | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| kernel-shap | taxi | algos | 377 | 1295 | shap-cpu | 920 | 0.41 | 1.41 | FLIP slower | rel_error_vs_exact=1.73139e-08 | - | rel_error_vs_exact=6.394e-14 | ok |
| mb-dict-learning | istella | algos | 8686 | 7069 | sklearn-cpu | 5030 | 1.73 | 1.41 |  | component_sparsity=0.0863636, relative_reconstruction_error=0.64834 | - | component_sparsity=0.08636, relative_reconstruction_error=0.644 | ok |
| ivf-filter | istella | algos | 10184 | 10114 | faiss-cpu | 7234 | 1.41 | 1.40 |  | recall_at_10=0.60925 | - | recall_at_10=0.8419 | ok |
| select-mutual-info-reg | istella | algos | 69680 | 69680 | sklearn-cpu | 50021 | 1.39 | 1.39 |  | n_selected=110 | n_selected=110 | jaccard_vs_sklearn=1, n_selected=110 | ok |
| ivf-refine | istella | algos | 10101 | 10124 | faiss-cpu | 7322 | 1.38 | 1.38 |  | recall_at_10=0.809175 | - | recall_at_10=0.9934 | ok |
| louvain | istella | algos | 764 | 1880 | networkx-cpu | 1376 | 0.55 | 1.37 | FLIP slower | modularity=0.911187, n_communities=40 | - | modularity=0.9085, n_communities=40 | ok |
| knn | istella | classical | 768 | 766 | sklearn-cpu | 565 | 1.36 | 1.35 |  | recall_at_k=0.97625, rows_with_repeated_ids=0 | - | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| perceptron | taxi | algos | 3735 | 1433 | sklearn-cpu | 1081 | 3.46 | 1.33 |  | accuracy=0.76219 | - | accuracy=0.7505 | ok |
| knn-imputer | taxi | algos | 70.9 | 2.2 | sklearn-cpu | 1.7 | 42.35 | 1.32 |  | masked_rmse=6.1517 | - | masked_rmse=5.257 | ok |
| resample | taxi | algos | 66.3 | 68.5 | sklearn-cpu | 51.9 | 1.28 | 1.32 |  | max_mean_shift_over_std=0.0029166 | - | max_mean_shift_over_std=0.002257 | ok |
| connected-components | taxi | algos | 84.0 | 9.5 | networkx-cpu | 7.3 | 11.43 | 1.29 |  | n_components=588 | - | n_components=588 | ok |
| pa-reg | taxi | algos | 3785 | 1511 | sklearn-cpu | 1182 | 3.20 | 1.28 |  | finite=1, r2=0.900533, rmse=5.02315 | - | r2=0.7951, rmse=7.209 | ok |
| select-d | synthetic | algos | 5.5 | 6.4 | statsmodels-cpu | 5.1 | 1.09 | 1.27 |  | - | - | d_agreement_vs_statsmodels=1 | ok |
| randomized-svd | taxi | algos | 194 | 206 | sklearn-cpu | 163 | 1.19 | 1.26 |  | relative_reconstruction_error=0.0271968 | - | relative_reconstruction_error=0.0272 | ok |
| lasso-lars | istella | algos | 1064 | 287 | sklearn-cpu | 228 | 4.66 | 1.26 |  | finite=1, r2=0.31033, rmse=0.693715 | - | r2=0.3111, rmse=0.6933 | ok |
| samba-forward | bytes | neural | 55.8 | 55.8 | torch-eager-fp32 | 44.5 | 1.25 | 1.25 |  | mean_nll=5.636 | mean_nll=5.636 | max_abs_diff_vs_ours=2.503e-06, max_rel_diff_vs_ours=1.407e-06, mean_nll=5.636 | ok |
| pa-clf | taxi | algos | 3747 | 1503 | sklearn-cpu | 1215 | 3.08 | 1.24 |  | accuracy=0.76036 | - | accuracy=0.7447 | ok |
| maxabs-scaler | istella | algos | 228 | 122 | sklearn-cpu | 99.7 | 2.29 | 1.22 |  | - | - | - | ok |
| spectral | taxi | classical2 | 468 | 1124 | sklearn-cpu | 923 | 0.51 | 1.22 | FLIP slower | n_clusters=8, silhouette=0.0399104 | - | ari_vs_ours=0.5823, n_clusters=8, silhouette=0.08989 | ok |
| knn-imputer | istella | algos | - | 31.7 | sklearn-cpu (fill) | 26.0 | - | 1.22 |  | masked_rmse=323953 | - | masked_rmse=9.78e+05 | ok |
| dart-reg | istella | algos | 61244 | 39168 | lightgbm-cpu | 32389 | 1.89 | 1.21 |  | finite=1, r2=0.550733, rmse=0.559903 | - | r2=0.5647, rmse=0.5512 | ok |
| multilabel-binarizer | istella | algos | 783 | 173 | sklearn-cpu | 143 | 5.45 | 1.20 |  | - | - | - | ok |
| nystroem | istella | classical2 | 570 | 610 | sklearn-cpu | 509 | 1.12 | 1.20 |  | kernel_rel_error=0.0334304 | - | kernel_rel_error=0.03896 | ok |
| auto-theta | taxi-hourly | algos | 4756 | 4770 | statsforecast-cpu | 4032 | 1.18 | 1.18 |  | forecast_rmse=49.0546 | - | forecast_rmse=49.27 | ok |
| gmm | istella | classical2 | 9182 | 9155 | sklearn-cpu | 7866 | 1.17 | 1.16 |  | bic=-3.85139e+07, mean_log_likelihood=200.794, n_iter=24 | - | bic=-3.901e+07, mean_log_likelihood=200.8, n_iter=30 | ok |
| resample | istella | algos | 392 | 390 | sklearn-cpu | 336 | 1.17 | 1.16 |  | max_mean_shift_over_std=0.00320274 | - | max_mean_shift_over_std=0.002552 | ok |
| theta | synthetic | algos | 120 | 120 | statsmodels-cpu | 104 | 1.16 | 1.16 |  | forecast_rmse=1.43661 | - | forecast_rmse=1.435 | ok |
| incremental-pca | taxi | algos | 535 | 160 | sklearn-cpu | 138 | 3.88 | 1.16 |  | explained_variance_fraction=0.999995 | - | explained_variance_fraction=1 | ok |
| select-f-classif | istella | algos | 344 | 336 | sklearn-cpu | 294 | 1.17 | 1.14 |  | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| select-chi2 | istella | algos | 345 | 318 | sklearn-cpu | 283 | 1.22 | 1.12 |  | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| ridge-clf | taxi | algos | 379 | 86.9 | sklearn-cpu | 77.9 | 4.86 | 1.12 |  | accuracy=0.76357 | - | accuracy=0.7636 | ok |
| dart | istella | algos | 61578 | 39525 | lightgbm-cpu | 35425 | 1.74 | 1.12 |  | accuracy=0.94872, logloss=0.134126 | - | accuracy=0.9519, logloss=0.1237 | ok |
| ols | taxi | classical | 115 | 303 | sklearn-cpu | 274 | 0.42 | 1.10 | FLIP slower | finite=1, r2=0.908837, rmse=4.69648 | - | r2=0.7248, rmse=8.159 | ok |
| label-propagation | istella | algos | - | 15655 | sklearn-cpu | 14552 | - | 1.08 |  | accuracy=0.9055 | - | accuracy=0.9055 | ok |
| lof | istella | algos | - | 10899 | sklearn-cpu | 10177 | - | 1.07 |  | fraction_flagged=0.03361 | - | fraction_flagged=0.03361, jaccard_vs_sklearn=1 | ok |
| nmf | istella | algos | 9487 | 7483 | sklearn-cpu | 7059 | 1.34 | 1.06 |  | relative_reconstruction_error=0.325174 | - | relative_reconstruction_error=0.3254 | ok |
| quantile-transformer | taxi | algos | 183 | 184 | sklearn-cpu | 174 | 1.05 | 1.06 |  | - | - | - | ok |
| label-spreading | istella | algos | - | 10856 | sklearn-cpu | 10273 | - | 1.06 |  | accuracy=0.90445 | - | accuracy=0.9044 | ok |
| select-mutual-info-reg | taxi | algos | 2795 | 2796 | sklearn-cpu | 2715 | 1.03 | 1.03 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| ivf | istella | classical2 | 4815 | 5065 | faiss-cpu | 5025 | 0.96 | 1.01 | FLIP slower | - | - | - | ok |
| select-f-regression | istella | algos | 266 | 269 | sklearn-cpu | 268 | 0.99 | 1.00 | FLIP slower | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| ard | taxi | algos | 51.7 | 17.4 | sklearn-cpu | 17.5 | 2.95 | 1.00 | FLIP faster | finite=1, r2=0.909193, rmse=4.79951 | - | r2=0.9092, rmse=4.8 | ok |
| svd | taxi | algos | 67.3 | 51.5 | torch-gpu | 51.8 | 1.30 | 1.00 | FLIP faster | max_rel_singular_value_error=9.29813e-07, relative_reconstruction_error_100k_rows=1.82966e-06 | - | max_rel_singular_value_error=2.225e-06, relative_reconstruction_error_100k_rows=1.351e-05 | ok |
| select-r-regression | istella | algos | 282 | 264 | sklearn-cpu | 268 | 1.05 | 0.98 | FLIP faster | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| ols | istella | classical | 1205 | 3070 | sklearn-cpu | 3245 | 0.37 | 0.95 |  | finite=1, r2=0.332506, rmse=0.68174 | - | r2=0.001881, rmse=0.8337 | ok |
| cross-val-score | istella | algos | 4097 | 6667 | sklearn-cpu | 7051 | 0.58 | 0.95 |  | mean_r2=0.334732 | - | mean_r2=0.3198 | ok |
| umap | taxi | classical2 | 577 | 1320 | umap-learn-cpu-unseeded | 1434 | 0.40 | 0.92 |  | trustworthiness_k15=0.991778 | - | trustworthiness_k15=0.9893 | ok |
| svc | taxi | classical | 737 | 2269 | sklearn-cpu | 2467 | 0.30 | 0.92 |  | accuracy=0.7675, n_support=5527 | - | accuracy=0.7675, n_support=5672 | ok |
| prophet | synthetic | algos | 377 | 414 | prophet-cpu | 452 | 0.83 | 0.92 |  | forecast_rmse=1.01515 | - | forecast_rmse=1.015 | ok |
| ivf-sq | istella | algos | 4475 | 4534 | faiss-cpu | 5135 | 0.87 | 0.88 |  | recall_at_10=0.728025 | - | recall_at_10=0.5913 | ok |
| mlp-reg | taxi | algos | 8701 | 8701 | sklearn-cpu | 9959 | 0.87 | 0.87 |  | r2=0.932, rmse=4.154 | r2=0.932, rmse=4.154 | r2=0.9296, rmse=4.226 | ok |
| gpr | taxi | classical2 | 236 | 141 | sklearn-cpu | 164 | 1.44 | 0.86 | FLIP faster | finite=1, mean_log_predictive_density=-311.458, r2=0.88963, rmse=5.04164 | - | mean_log_predictive_density=-311.5, r2=0.8896, rmse=5.042 | ok |
| mlp-reg | istella | algos | 11542 | 11542 | sklearn-cpu | 13651 | 0.85 | 0.85 |  | r2=0.5264, rmse=0.5749 | r2=0.5264, rmse=0.5749 | r2=0.5248, rmse=0.5758 | ok |
| kernel-ridge | taxi | classical2 | 668 | 652 | sklearn-cpu | 794 | 0.84 | 0.82 |  | finite=1, r2=0.726543, rmse=8.33037 | - | r2=0.7265, rmse=8.33 | ok |
| louvain | taxi | algos | 722 | 636 | networkx-cpu | 781 | 0.92 | 0.81 |  | modularity=0.941953, n_communities=58 | - | modularity=0.9408, n_communities=56 | ok |
| gpc | taxi | classical2 | 589 | 589 | sklearn-cpu | 729 | 0.81 | 0.81 |  | accuracy=0.761, logloss=0.5413, nonfinite_proba_rows=0 | accuracy=0.761, logloss=0.5413, nonfinite_proba_rows=0 | accuracy=0.761, logloss=0.5414, nonfinite_proba_rows=0 | ok |
| mlp-clf | istella | algos | 11568 | 11568 | sklearn-cpu | 14327 | 0.81 | 0.81 |  | accuracy=0.9443, logloss=0.137 | accuracy=0.9443, logloss=0.137 | accuracy=0.9438, logloss=0.1368 | ok |
| svd | istella | algos | 17796 | 1974 | torch-gpu | 2473 | 7.20 | 0.80 | FLIP faster | max_rel_singular_value_error=41531.1, relative_reconstruction_error_100k_rows=3.84493e-05 | - | max_rel_singular_value_error=4.945e+06, relative_reconstruction_error_100k_rows=0.0005905 | ok |
| complement-nb | text | algos | 244 | 208 | sklearn-cpu | 265 | 0.92 | 0.79 |  | accuracy=0.983067, logloss=0.559491 | - | accuracy=0.9831, logloss=0.5573 | ok |
| ivf-rabitq | istella | algos | 4268 | 4292 | faiss-cpu | 5474 | 0.78 | 0.78 |  | recall_at_10=0.125125 | - | recall_at_10=0.05045 | ok |
| multinomial-nb | text | algos | 248 | 206 | sklearn-cpu | 266 | 0.93 | 0.78 |  | accuracy=0.983067, logloss=0.559529 | - | accuracy=0.9831, logloss=0.5573 | ok |
| enet-cv | taxi | algos | 3247 | 163 | sklearn-cpu | 211 | 15.37 | 0.77 | FLIP faster | finite=1, r2=0.909002, rmse=4.80454 | - | r2=0.909, rmse=4.804 | ok |
| connected-components | istella | algos | 96.2 | 5.4 | networkx-cpu | 7.0 | 13.82 | 0.77 | FLIP faster | n_components=81 | - | n_components=81 | ok |
| bisecting-kmeans | taxi | algos | 399 | 296 | sklearn-cpu | 383 | 1.04 | 0.77 | FLIP faster | n_clusters=8, silhouette=0.155357 | - | ari_vs_ours=0.5023, n_clusters=8, silhouette=0.128 | ok |
| rnn-reg | taxi-hourly | algos | 1571 | 1571 | torch-eager-fp32 (fill) | 2049 | 0.77 | 0.77 |  | r2=0.7388, rmse=0.5543 | r2=0.7388, rmse=0.5543 | - | ok |
| mlp-clf | taxi | algos | 8718 | 8718 | sklearn-cpu | 11431 | 0.76 | 0.76 |  | accuracy=0.7678, logloss=0.5305 | accuracy=0.7678, logloss=0.5305 | accuracy=0.7678, logloss=0.5304 | ok |
| kernel-ridge | istella | classical2 | 650 | 637 | sklearn-cpu | 836 | 0.78 | 0.76 |  | finite=1, r2=0.385427, rmse=0.646407 | - | r2=0.3854, rmse=0.6464 | ok |
| rnn-reg | synthetic | algos | 1540 | 1540 | torch-eager-fp32 (fill) | 2037 | 0.76 | 0.76 |  | r2=0.9774, rmse=0.1744 | r2=0.9774, rmse=0.1744 | - | ok |
| iterative-imputer | istella | algos | 7815 | 7808 | sklearn-cpu | 10351 | 0.75 | 0.75 |  | masked_rmse=799013 | - | masked_rmse=8.024e+05 | ok |
| rnn-clf | synthetic | algos | 1568 | 1568 | torch-eager-fp32 (fill) | 2088 | 0.75 | 0.75 |  | accuracy=0.9536, logloss=0.1037 | accuracy=0.9536, logloss=0.1037 | - | ok |
| label-encoder | istella | algos | 293 | 16.9 | sklearn-cpu | 22.5 | 12.99 | 0.75 | FLIP faster | - | - | - | ok |
| perceptron | istella | algos | 34349 | 4294 | sklearn-cpu | 5738 | 5.99 | 0.75 | FLIP faster | accuracy=0.89569 | - | accuracy=0.8961 | ok |
| multinomial-nb | istella | algos | 339 | 136 | sklearn-cpu | 183 | 1.85 | 0.74 | FLIP faster | accuracy=0.85362, logloss=3.62857 | - | accuracy=0.8536, logloss=3.087 | ok |
| rnn-clf | taxi-hourly | algos | 1529 | 1529 | torch-eager-fp32 (fill) | 2087 | 0.73 | 0.73 |  | accuracy=0.8681, logloss=0.3049 | accuracy=0.8681, logloss=0.3049 | - | ok |
| pca | taxi | classical | 96.8 | 86.6 | sklearn-cpu | 120 | 0.81 | 0.72 |  | explained_variance_ratio_sum=0.999996 | - | explained_variance_ratio_sum=1 | ok |
| complement-nb | istella | algos | 338 | 135 | sklearn-cpu | 186 | 1.82 | 0.72 | FLIP faster | accuracy=0.84936, logloss=3.76252 | - | accuracy=0.8494, logloss=3.175 | ok |
| prophet | taxi-hourly | algos | 420 | 381 | prophet-cpu | 549 | 0.76 | 0.69 |  | forecast_rmse=32.0493 | - | forecast_rmse=32.03 | ok |
| lasso-cv | taxi | algos | 3242 | 156 | sklearn-cpu | 226 | 14.32 | 0.69 | FLIP faster | finite=1, r2=0.909059, rmse=4.80305 | - | r2=0.909, rmse=4.804 | ok |
| mb-dict-learning | taxi | algos | 3650 | 3639 | sklearn-cpu | 5293 | 0.69 | 0.69 |  | component_sparsity=0, relative_reconstruction_error=0.496686 | - | component_sparsity=0, relative_reconstruction_error=0.477 | ok |
| minmax-scaler | taxi | algos | 21.2 | 11.8 | sklearn-cpu | 17.6 | 1.21 | 0.67 | FLIP faster | - | - | - | ok |
| ocsvm | istella | algos | 476 | 463 | sklearn-cpu | 691 | 0.69 | 0.67 |  | fraction_flagged=0.0783 | - | fraction_flagged=0.0783, jaccard_vs_sklearn=1 | ok |
| dynamic-theta | taxi-hourly | algos | 511 | 511 | statsforecast-cpu | 768 | 0.67 | 0.67 |  | forecast_rmse=49.1012 | - | forecast_rmse=49.27 | ok |
| label-propagation | taxi | algos | 8430 | 6835 | sklearn-cpu | 10366 | 0.81 | 0.66 |  | accuracy=0.7016 | - | accuracy=0.7016 | ok |
| dynamic-optimized-theta | synthetic | algos | 623 | 625 | statsforecast-cpu | 959 | 0.65 | 0.65 |  | forecast_rmse=1.43628 | - | forecast_rmse=1.436 | ok |
| gpc | istella | classical2 | 798 | 798 | sklearn-cpu | 1230 | 0.65 | 0.65 |  | accuracy=0.9013, logloss=0.2326, nonfinite_proba_rows=0 | accuracy=0.9013, logloss=0.2326, nonfinite_proba_rows=0 | accuracy=0.9013, logloss=0.2326, nonfinite_proba_rows=0 | ok |
| lof | taxi | algos | 2782 | 2209 | sklearn-cpu | 3405 | 0.82 | 0.65 |  | fraction_flagged=0.00896 | - | fraction_flagged=0.00896, jaccard_vs_sklearn=1 | ok |
| maxabs-scaler | taxi | algos | 121 | 8.0 | sklearn-cpu | 12.4 | 9.76 | 0.65 | FLIP faster | - | - | - | ok |
| bayesian-gmm | taxi | algos | 2544 | 2506 | sklearn-cpu | 3978 | 0.64 | 0.63 |  | mean_log_likelihood=4.8956 | - | mean_log_likelihood=6.178 | ok |
| auto-theta | synthetic | algos | 1226 | 1225 | statsforecast-cpu | 1953 | 0.63 | 0.63 |  | forecast_rmse=1.43891 | - | forecast_rmse=1.438 | ok |
| kmeans | istella | classical | 2182 | 2072 | sklearn-cpu | 3354 | 0.65 | 0.62 |  | inertia=6.05072e+17, inertia_over_ours=1, n_iter=33 | - | inertia=5.959e+17, inertia_over_ours=0.9848, n_iter=36 | ok |
| als | taxi-zones | algos | - | 18651 | implicit-cpu | 30318 | - | 0.62 |  | recall_at_10=0.0539953 | - | recall_at_10=0.05679 | ok |
| spectral | istella | classical2 | 372 | 961 | sklearn-cpu | 1595 | 0.23 | 0.60 |  | n_clusters=8, silhouette=0.147668 | - | ari_vs_ours=0.9998, n_clusters=8, silhouette=0.1477 | ok |
| robust-scaler | taxi | algos | 182 | 65.7 | sklearn-cpu | 109 | 1.67 | 0.60 | FLIP faster | - | - | - | ok |
| knn-clf | istella | classical2 | 213 | 171 | sklearn-cpu | 293 | 0.73 | 0.58 |  | accuracy=0.92625 | - | accuracy=0.9263 | ok |
| knn-reg | istella | classical2 | 195 | 164 | sklearn-cpu | 285 | 0.68 | 0.58 |  | finite=1, r2=0.418145, rmse=0.625388 | - | r2=0.4181, rmse=0.6254 | ok |
| gru-reg | synthetic | algos | 1720 | 1720 | torch-eager-fp32 (fill) | 3060 | 0.56 | 0.56 |  | r2=0.9819, rmse=0.1557 | r2=0.9819, rmse=0.1557 | - | ok |
| gru-reg | taxi-hourly | algos | 1727 | 1727 | torch-eager-fp32 (fill) | 3072 | 0.56 | 0.56 |  | r2=0.7482, rmse=0.5442 | r2=0.7482, rmse=0.5442 | - | ok |
| gru-clf | taxi-hourly | algos | 1722 | 1722 | torch-eager-fp32 (fill) | 3099 | 0.56 | 0.56 |  | accuracy=0.8657, logloss=0.3058 | accuracy=0.8657, logloss=0.3058 | - | ok |
| tree-shap | istella | algos | 846 | 81.9 | lightgbm-cpu | 148 | 5.71 | 0.55 | FLIP faster | max_additivity_error=1.17498e-06 | - | max_additivity_error=4.441e-15 | ok |
| gru-clf | synthetic | algos | 1725 | 1725 | torch-eager-fp32 (fill) | 3137 | 0.55 | 0.55 |  | accuracy=0.9718, logloss=0.06584 | accuracy=0.9718, logloss=0.06584 | - | ok |
| optimized-theta | taxi-hourly | algos | 708 | 708 | statsforecast-cpu | 1290 | 0.55 | 0.55 |  | forecast_rmse=49.1509 | - | forecast_rmse=49.36 | ok |
| nearest-centroid | istella | algos | 6456 | 289 | sklearn-cpu | 529 | 12.19 | 0.55 | FLIP faster | accuracy=0.85261, logloss=4.29922 | - | accuracy=0.8526, logloss=4.118 | ok |
| poly-count-sketch | taxi | algos | 0.3 | 0.3 | sklearn-cpu | 0.5 | 0.52 | 0.54 |  | kernel_rel_error=0.0965957 | - | kernel_rel_error=0.0966 | ok |
| lstsq | taxi | algos | 32.0 | 28.3 | numpy-cpu | 52.2 | 0.61 | 0.54 |  | relative_residual=0.756366 | - | relative_residual=0.7564 | ok |
| meanshift | istella | algos | 6607 | 237 | sklearn-cpu | 444 | 14.87 | 0.53 | FLIP faster | n_clusters=12, silhouette=0.403452 | - | ari_vs_ours=1, n_clusters=12, silhouette=0.4034 | ok |
| optimized-theta | synthetic | algos | 397 | 397 | statsforecast-cpu | 749 | 0.53 | 0.53 |  | forecast_rmse=1.43886 | - | forecast_rmse=1.438 | ok |
| incremental-pca | istella | algos | 2905 | 2905 | sklearn-cpu | 5745 | 0.51 | 0.51 |  | explained_variance_fraction=1 | explained_variance_fraction=1 | explained_variance_fraction=1 | error |
| lda | text | algos | - | 36638 | sklearn-cpu | 72657 | - | 0.50 |  | perplexity=266.719 | - | perplexity=266.9 | ok |
| simple-imputer | taxi | algos | 181 | 184 | sklearn-cpu | 370 | 0.49 | 0.50 |  | masked_rmse=5.98518 | - | masked_rmse=5.985 | ok |
| isotonic | taxi | algos | 1115 | 44.8 | sklearn-cpu | 90.5 | 12.33 | 0.50 | FLIP faster | finite=1, r2=0.897069, rmse=5.10987 | - | r2=0.8971, rmse=5.11 | ok |
| pa-clf | istella | algos | 63531 | 4399 | sklearn-cpu | 9215 | 6.89 | 0.48 | FLIP faster | accuracy=0.92259 | - | accuracy=0.8905 | ok |
| dict-learning | taxi | algos | 8869 | 5472 | sklearn-cpu | 11561 | 0.77 | 0.47 |  | component_sparsity=0, relative_reconstruction_error=0.459169 | - | component_sparsity=0, relative_reconstruction_error=0.4606 | ok |
| kmeans | taxi | classical | 960 | 951 | sklearn-cpu | 2058 | 0.47 | 0.46 |  | inertia=3.09303e+08, inertia_over_ours=1, n_iter=91 | - | inertia=3.166e+08, inertia_over_ours=1.024, n_iter=100 | ok |
| pls-canonical | taxi | algos | 119 | 123 | sklearn-cpu | 269 | 0.44 | 0.46 |  | mean_canonical_corr=0.559206 | - | mean_canonical_corr=0.5592 | ok |
| ridge-clf | istella | algos | 8384 | 3207 | sklearn-cpu | 7044 | 1.19 | 0.46 | FLIP faster | accuracy=0.91054 | - | accuracy=0.9105 | ok |
| dict-learning | istella | algos | 17160 | 7442 | sklearn-cpu | 16442 | 1.04 | 0.45 | FLIP faster | component_sparsity=0.0863636, relative_reconstruction_error=0.652121 | - | component_sparsity=0.08636, relative_reconstruction_error=0.6521 | ok |
| pls | taxi | algos | 116 | 108 | sklearn-cpu | 242 | 0.48 | 0.45 |  | finite=1, r2=0.905216, rmse=4.90347 | - | r2=0.9052, rmse=4.904 | ok |
| mb-sparse-pca | istella | algos | 2398 | 877 | sklearn-cpu | 2006 | 1.20 | 0.44 | FLIP faster | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | component_sparsity=0.1307, relative_reconstruction_error=0.7054 | ok |
| random-trees-embedding | istella | algos | 271 | 270 | sklearn-cpu | 625 | 0.43 | 0.43 |  | nonzeros_per_row=10, output_columns=209 | - | nonzeros_per_row=10, output_columns=251 | ok |
| complement-nb | taxi | algos | 210 | 17.4 | sklearn-cpu | 40.9 | 5.13 | 0.43 | FLIP faster | accuracy=0.67802, logloss=0.715492 | - | accuracy=0.678, logloss=0.7155 | ok |
| dynamic-theta | synthetic | algos | 139 | 139 | statsforecast-cpu | 333 | 0.42 | 0.42 |  | forecast_rmse=1.43726 | - | forecast_rmse=1.437 | ok |
| multinomial-nb | taxi | algos | 209 | 16.7 | sklearn-cpu | 40.8 | 5.13 | 0.41 | FLIP faster | accuracy=0.72316, logloss=0.590725 | - | accuracy=0.7232, logloss=0.5907 | ok |
| bpe-encode | enwik8 | algos | 58.6 | 53.8 | hf-tokenizers-cpu | 134 | 0.44 | 0.40 |  | documents_equal_to_ours=1, tokens=1.45732e+06 | - | documents_equal_to_ours=1, tokens=1.457e+06 | ok |
| als | text | algos | - | 127744 | implicit-cpu (fill) | 321338 | - | 0.40 |  | recall_at_10=0.548738 | - | recall_at_10=0.5482 | ok |
| cca | taxi | algos | 179 | 184 | sklearn-cpu | 464 | 0.39 | 0.40 |  | mean_canonical_corr=0.576863 | - | mean_canonical_corr=0.5769 | ok |
| label-encoder | taxi | algos | 300 | 11.5 | sklearn-cpu | 29.0 | 10.33 | 0.40 | FLIP faster | - | - | - | ok |
| gaussian-nb | istella | algos | 405 | 141 | sklearn-cpu | 361 | 1.12 | 0.39 | FLIP faster | accuracy=0.87657, logloss=3.57442 | - | accuracy=0.8765, logloss=3.417 | ok |
| lasso-cv | istella | algos | 76022 | 2125 | sklearn-cpu | 5483 | 13.87 | 0.39 | FLIP faster | finite=1, r2=0.310329, rmse=0.693715 | - | r2=0.3108, rmse=0.6935 | ok |
| pls-canonical | istella | algos | 2701 | 1560 | sklearn-cpu | 4065 | 0.66 | 0.38 |  | mean_canonical_corr=0.87534 | - | mean_canonical_corr=0.8753 | ok |
| kbins | istella | algos | 1440 | 1423 | sklearn-cpu | 3823 | 0.38 | 0.37 |  | - | - | - | ok |
| qr | taxi | algos | 197 | 42.3 | numpy-cpu | 114 | 1.72 | 0.37 | FLIP faster | relative_gram_difference=5.57879e-07 | - | relative_gram_difference=3.024e-08 | ok |
| mamba3-forward | gaussian | neural | 29.0 | 29.0 | torch-eager-fp32 | 79.9 | 0.36 | 0.36 |  | - | - | max_abs_diff_vs_ours=8.345e-07, max_rel_diff_vs_ours=3.651e-07 | ok |
| arima | synthetic | classical2 | 95.7 | 132 | statsmodels-cpu | 364 | 0.26 | 0.36 |  | forecast_rmse=1.51552, insample_rmse=0.999342, mean_aic=5680.98, mean_llf=-2836.49 | - | forecast_rmse=1.515, insample_rmse=0.9993, mean_aic=5681, mean_llf=-2836 | ok |
| pa-reg | istella | algos | 65446 | 4406 | sklearn-cpu | 12345 | 5.30 | 0.36 | FLIP faster | finite=1, r2=0.294299, rmse=0.701731 | - | r2=-0.1282, rmse=0.8872 | ok |
| ivf-refine | taxi | algos | 1535 | 1535 | faiss-cpu | 4363 | 0.35 | 0.35 |  | recall_at_10=0.9997 | recall_at_10=0.9997 | recall_at_10=0.9992 | ok |
| ivf-pq | taxi | algos | 1576 | 1533 | faiss-cpu | 4367 | 0.36 | 0.35 |  | recall_at_10=0.973225 | - | recall_at_10=0.9801 | ok |
| ivf-filter | taxi | algos | 1586 | 1519 | faiss-cpu | 4368 | 0.36 | 0.35 |  | recall_at_10=0.980075 | - | recall_at_10=0.982 | ok |
| variance-threshold | istella | algos | 260 | 239 | sklearn-cpu | 688 | 0.38 | 0.35 |  | - | - | - | ok |
| fastica | taxi | algos | 123 | 145 | sklearn-cpu | 419 | 0.29 | 0.35 |  | mean_abs_excess_kurtosis=13.2051 | - | mean_abs_excess_kurtosis=13.77 | ok |
| sparse-pca | istella | algos | - | 3310 | sklearn-cpu | 9904 | - | 0.33 |  | component_sparsity=0.305682, relative_reconstruction_error=0.75021 | - | component_sparsity=0.3057, relative_reconstruction_error=0.7502 | ok |
| rfe | taxi | algos | 130 | 100 | sklearn-cpu | 313 | 0.42 | 0.32 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| tree-shap | taxi | algos | 345 | 23.2 | lightgbm-cpu | 72.7 | 4.75 | 0.32 | FLIP faster | max_additivity_error=3.85772e-05 | - | max_additivity_error=5.684e-13 | ok |
| agglomerative | taxi | classical2 | 71.1 | 71.2 | sklearn-cpu | 224 | 0.32 | 0.32 |  | n_clusters=8, silhouette=0.685524 | - | ari_vs_ours=1, n_clusters=8, silhouette=0.6855 | ok |
| gpr | istella | classical2 | 259 | 167 | sklearn-cpu | 523 | 0.49 | 0.32 |  | finite=1, mean_log_predictive_density=-9.28575, r2=0.235346, rmse=0.760439 | - | mean_log_predictive_density=-9.287, r2=0.2354, rmse=0.7604 | ok |
| tsvd | istella | classical2 | 424 | 447 | sklearn-cpu | 1408 | 0.30 | 0.32 |  | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.00255392 | - | explained_variance_ratio_sum=1, relative_reconstruction_error=0.000122 | ok |
| label-spreading | taxi | algos | 4719 | 2058 | sklearn-cpu | 6648 | 0.71 | 0.31 |  | accuracy=0.6764 | - | accuracy=0.6764 | ok |
| isomap | istella | algos | - | 4088 | sklearn-cpu | 13252 | - | 0.31 |  | trustworthiness_k15=0.853294 | - | trustworthiness_k15=0.8533 | ok |
| pls | istella | algos | 621 | 622 | sklearn-cpu | 2018 | 0.31 | 0.31 |  | finite=1, r2=0.28987, rmse=0.70393 | - | r2=0.2899, rmse=0.7039 | ok |
| enet-cv | istella | algos | 80048 | 2122 | sklearn-cpu | 6979 | 11.47 | 0.30 | FLIP faster | finite=1, r2=0.316583, rmse=0.690563 | - | r2=0.3173, rmse=0.6902 | ok |
| lstsq | istella | algos | 5860 | 915 | numpy-cpu | 3086 | 1.90 | 0.30 | FLIP faster | relative_residual=0.849956 | - | relative_residual=0.8733 | ok |
| robust-scaler | istella | algos | 1508 | 1310 | sklearn-cpu | 4588 | 0.33 | 0.29 |  | - | - | - | ok |
| standard-scaler | taxi | algos | 26.2 | 12.8 | sklearn-cpu | 45.4 | 0.58 | 0.28 |  | - | - | - | ok |
| quantile-transformer | istella | algos | 1523 | 1424 | sklearn-cpu | 5049 | 0.30 | 0.28 |  | - | - | - | ok |
| standard-scaler | istella | algos | 314 | 134 | sklearn-cpu | 478 | 0.66 | 0.28 |  | - | - | - | ok |
| dart | taxi | algos | 10482 | 7367 | lightgbm-cpu | 26498 | 0.40 | 0.28 |  | accuracy=0.76834, logloss=0.52906 | - | accuracy=0.7682, logloss=0.5291 | ok |
| bayesian-gmm | istella | algos | - | 32468 | sklearn-cpu (fill) | 116803 | - | 0.28 |  | mean_log_likelihood=175.322 | - | - | ok |
| factor-analysis | istella | algos | - | 9521 | sklearn-cpu | 35135 | - | 0.27 |  | mean_log_likelihood=99.4872 | - | mean_log_likelihood=98.12 | ok |
| nmf | taxi | algos | 2237 | 863 | sklearn-cpu | 3419 | 0.65 | 0.25 |  | relative_reconstruction_error=0.0911559 | - | relative_reconstruction_error=0.09115 | ok |
| dart-reg | taxi | algos | 9783 | 6880 | lightgbm-cpu | 27650 | 0.35 | 0.25 |  | finite=1, r2=0.925497, rmse=4.34734 | - | r2=0.9263, rmse=4.325 | ok |
| gaussian-nb | taxi | algos | 306 | 18.9 | sklearn-cpu | 76.2 | 4.01 | 0.25 | FLIP faster | accuracy=0.71982, logloss=1.13225 | - | accuracy=0.7199, logloss=1.134 | ok |
| cross-val-score | taxi | algos | 148 | 317 | sklearn-cpu | 1282 | 0.12 | 0.25 |  | mean_r2=0.937956 | - | mean_r2=0.938 | ok |
| categorical-nb | taxi | algos | 327 | 18.8 | sklearn-cpu | 78.3 | 4.17 | 0.24 | FLIP faster | accuracy=0.76585, logloss=0.538866 | - | accuracy=0.7659, logloss=0.5389 | ok |
| ridge-cv | istella | algos | - | 31606 | sklearn-cpu | 134463 | - | 0.24 |  | finite=1, r2=0.328684, rmse=0.684422 | - | r2=0.3287, rmse=0.6844 | ok |
| isomap | taxi | algos | - | 3202 | sklearn-cpu | 13665 | - | 0.23 |  | trustworthiness_k15=0.771828 | - | trustworthiness_k15=0.7718 | ok |
| spectral-embedding | taxi | classical2 | 228 | 516 | sklearn-cpu | 2243 | 0.10 | 0.23 |  | trustworthiness_k15=0.884879 | - | trustworthiness_k15=0.898 | ok |
| mamba1-forward | gaussian | neural | 21.2 | 21.2 | torch-eager-fp32 | 94.7 | 0.22 | 0.22 |  | - | - | max_abs_diff_vs_ours=2.384e-07, max_rel_diff_vs_ours=1.189e-07 | ok |
| multioutput-reg | istella | algos | 1760 | 2976 | sklearn-cpu | 13401 | 0.13 | 0.22 |  | r2=0.455327 | - | r2=0.4553 | ok |
| mamba2-forward | gaussian | neural | 33.9 | 33.9 | torch-compile-bf16 | 153 | 0.22 | 0.22 |  | - | - | max_abs_diff_vs_ours=0.007544, max_rel_diff_vs_ours=0.002565 | ok |
| ets | synthetic | classical2 | 217 | 188 | statsmodels-cpu | 860 | 0.25 | 0.22 |  | forecast_rmse=0.984392, insample_rmse=0.990971 | - | forecast_rmse=0.9844, insample_rmse=0.9918 | ok |
| affinity-prop | istella | algos | 1860 | 1247 | sklearn-cpu | 5734 | 0.32 | 0.22 |  | n_clusters=342, silhouette=0.0897635 | - | ari_vs_ours=1, n_clusters=342, silhouette=0.08976 | ok |
| ridge | istella | classical2 | 860 | 1447 | sklearn-cpu | 6717 | 0.13 | 0.22 |  | finite=1, r2=0.328682, rmse=0.684423 | - | r2=0.3287, rmse=0.6844 | ok |
| categorical-nb | istella | algos | 341 | 18.1 | sklearn-cpu | 84.4 | 4.04 | 0.21 | FLIP faster | accuracy=0.83885, logloss=0.412625 | - | accuracy=0.8388, logloss=0.4126 | ok |
| bernoulli-nb | taxi | algos | 220 | 19.1 | sklearn-cpu | 92.5 | 2.37 | 0.21 | FLIP faster | accuracy=0.75556, logloss=0.557803 | - | accuracy=0.7556, logloss=0.5578 | ok |
| qr | istella | algos | 8294 | 1823 | numpy-cpu | 8840 | 0.94 | 0.21 |  | relative_gram_difference=1.53936e-07 | - | relative_gram_difference=2.473e-08 | ok |
| stacking-reg | istella | algos | - | 31640 | sklearn-cpu | 153421 | - | 0.21 |  | finite=1, r2=0.448395, rmse=0.620404 | - | r2=0.4476, rmse=0.6208 | ok |
| croston-optimized | synthetic | algos | 6.8 | 12.7 | statsforecast-cpu | 62.3 | 0.11 | 0.20 |  | forecast_rmse=1.67554 | - | forecast_rmse=1.675 | ok |
| power-transformer | taxi | algos | 2732 | 2698 | sklearn-cpu | 13228 | 0.21 | 0.20 |  | - | - | - | ok |
| tsvd | taxi | classical2 | 40.8 | 31.9 | sklearn-cpu | 157 | 0.26 | 0.20 |  | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.00325707 | - | explained_variance_ratio_sum=1, relative_reconstruction_error=0.003257 | ok |
| croston-optimized | taxi-hourly | algos | 11.8 | 12.2 | statsforecast-cpu | 60.5 | 0.19 | 0.20 |  | forecast_rmse=1.39825 | - | forecast_rmse=1.398 | ok |
| bernoulli-nb | istella | algos | 795 | 161 | sklearn-cpu | 869 | 0.91 | 0.19 |  | accuracy=0.79405, logloss=5.35063 | - | accuracy=0.7941, logloss=4.279 | ok |
| poisson | taxi | algos | - | 77.7 | sklearn-cpu (fill) | 442 | - | 0.18 |  | finite=1, r2=0.0357566, rmse=15.6398 | - | - | ok |
| knn-reg | taxi | classical2 | 27.3 | 22.5 | sklearn-cpu | 131 | 0.21 | 0.17 |  | finite=1, r2=0.937323, rmse=3.84203 | - | r2=0.9373, rmse=3.842 | ok |
| adaboost-reg | taxi | algos | 3144 | 2090 | sklearn-cpu | 12197 | 0.26 | 0.17 |  | finite=1, r2=0.679395, rmse=9.01824 | - | r2=0.5639, rmse=10.52 | ok |
| svr | taxi | classical2 | 114 | 195 | sklearn-cpu | 1144 | 0.10 | 0.17 |  | finite=1, r2=0.767551, rmse=7.6804 | - | r2=0.7675, rmse=7.68 | ok |
| tsne | istella | algos | 4903 | 3214 | sklearn-cpu | 19429 | 0.25 | 0.17 |  | trustworthiness_k15=0.992206 | - | trustworthiness_k15=0.992 | ok |
| sparse-pca | taxi | algos | 1964 | 1940 | sklearn-cpu | 11923 | 0.16 | 0.16 |  | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | component_sparsity=0.4886, relative_reconstruction_error=0.2772 | ok |
| voting-reg | istella | algos | - | 6096 | sklearn-cpu | 37765 | - | 0.16 |  | finite=1, r2=0.406568, rmse=0.643496 | - | r2=0.4067, rmse=0.6434 | ok |
| knn-clf | taxi | classical2 | 49.6 | 24.3 | sklearn-cpu | 154 | 0.32 | 0.16 |  | accuracy=0.74175 | - | accuracy=0.7418 | ok |
| poly-features | taxi | algos | 0.3 | 0.2 | sklearn-cpu | 1.6 | 0.17 | 0.15 |  | - | - | - | ok |
| ivf | taxi | classical2 | 588 | 560 | faiss-cpu | 3751 | 0.16 | 0.15 |  | - | - | - | ok |
| poly-features | istella | algos | 0.4 | 0.3 | sklearn-cpu | 2.1 | 0.21 | 0.15 |  | - | - | - | ok |
| lda | taxi-zones | algos | 41629 | 3065 | sklearn-cpu | 20926 | 1.99 | 0.15 | FLIP faster | perplexity=45.2218 | - | perplexity=44.9 | ok |
| gamma | taxi | algos | - | 52.7 | sklearn-cpu (fill) | 360 | - | 0.15 |  | finite=1, r2=-231.912, rmse=243.071 | - | - | ok |
| ivf-sq | taxi | algos | 523 | 527 | faiss-cpu | 3783 | 0.14 | 0.14 |  | recall_at_10=0.934975 | - | recall_at_10=0.857 | ok |
| croston-sba | synthetic | algos | 2.6 | 4.8 | statsforecast-cpu | 34.5 | 0.07 | 0.14 |  | forecast_rmse=1.67446 | - | forecast_rmse=1.675 | ok |
| lr-step | synthetic | algos | 11.8 | 11.8 | torch-cpu | 85.1 | 0.14 | 0.14 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=1.49e-08 | ok |
| cca | istella | algos | 18214 | 10056 | sklearn-cpu | 73061 | 0.25 | 0.14 |  | mean_canonical_corr=0.998053 | - | mean_canonical_corr=0.9996 | ok |
| svr | istella | classical2 | 110 | 195 | sklearn-cpu | 1422 | 0.08 | 0.14 |  | finite=1, r2=0.318258, rmse=0.680816 | - | r2=0.3182, rmse=0.6808 | ok |
| tweedie | taxi | algos | - | 45.1 | sklearn-cpu (fill) | 333 | - | 0.14 |  | finite=1, r2=-10.0744, rmse=53.0026 | - | - | ok |
| tsne | taxi | algos | 4014 | 2340 | sklearn-cpu | 17507 | 0.23 | 0.13 |  | trustworthiness_k15=0.99889 | - | trustworthiness_k15=0.9989 | ok |
| stacking-reg | taxi | algos | - | 794 | sklearn-cpu | 6007 | - | 0.13 |  | finite=1, r2=0.919718, rmse=4.51281 | - | r2=0.9325, rmse=4.137 | ok |
| logreg-cv | taxi | algos | - | 231 | sklearn-cpu | 1758 | - | 0.13 |  | accuracy=0.76332, logloss=0.538985 | - | accuracy=0.7633, logloss=0.539 | ok |
| ivf-rabitq | taxi | algos | 538 | 519 | faiss-cpu | 3984 | 0.13 | 0.13 |  | recall_at_10=0.110475 | - | recall_at_10=0.1263 | ok |
| svc | istella | classical | 108 | 142 | sklearn-cpu | 1092 | 0.10 | 0.13 |  | accuracy=0.9222, n_support=2400 | - | accuracy=0.9222, n_support=2400 | ok |
| hdbscan | taxi | classical | 4879 | 3928 | sklearn-cpu | 30810 | 0.16 | 0.13 |  | n_clusters=161, noise_fraction=0.14055, rows=100000 | - | ari_vs_ours=0.9912, n_clusters=160, noise_agreement_vs_ours=0.9983, noise_fraction=0.1422, | ok |
| linearsvc | taxi | classical2 | 109 | 56.3 | sklearn-cpu | 451 | 0.24 | 0.12 |  | accuracy=0.76333 | - | accuracy=0.7636 | ok |
| huber | taxi | algos | 20184 | 227 | sklearn-cpu | 1855 | 10.88 | 0.12 | FLIP faster | finite=1, r2=0.900215, rmse=5.03117 | - | r2=0.9002, rmse=5.031 | ok |
| select-mutual-info | taxi | algos | 218 | 218 | sklearn-cpu | 1786 | 0.12 | 0.12 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| rfe | istella | algos | 2217 | 1914 | sklearn-cpu | 16177 | 0.14 | 0.12 |  | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| poly-count-sketch | istella | algos | 0.4 | 0.4 | sklearn-cpu | 3.1 | 0.13 | 0.12 |  | kernel_rel_error=0.040849 | - | kernel_rel_error=0.04085 | ok |
| spectral-embedding | istella | classical2 | 446 | 955 | sklearn-cpu | 8259 | 0.05 | 0.12 |  | trustworthiness_k15=0.783644 | - | trustworthiness_k15=0.8127 | ok |
| simple-imputer | istella | algos | 1438 | 1425 | sklearn-cpu | 12578 | 0.11 | 0.11 |  | masked_rmse=346849 | - | masked_rmse=3.468e+05 | ok |
| bayesian-ridge | istella | algos | 101519 | 768 | sklearn-cpu | 6831 | 14.86 | 0.11 | FLIP faster | finite=1, r2=0.317909, rmse=0.689893 | - | r2=-890.2, rmse=24.94 | ok |
| random-trees-embedding | taxi | algos | 53.6 | 48.6 | sklearn-cpu | 441 | 0.12 | 0.11 |  | nonzeros_per_row=10, output_columns=292 | - | nonzeros_per_row=10, output_columns=244 | ok |
| voting-reg | taxi | algos | - | 132 | sklearn-cpu | 1272 | - | 0.10 |  | finite=1, r2=0.919181, rmse=4.52786 | - | r2=0.9246, rmse=4.372 | ok |
| multioutput-clf | taxi | algos | 223 | 67.2 | sklearn-cpu | 686 | 0.33 | 0.10 |  | accuracy=0.86356 | - | accuracy=0.8636 | ok |
| stacking-clf | taxi | algos | 2420 | 693 | sklearn-cpu | 7095 | 0.34 | 0.10 |  | accuracy=0.76792, logloss=0.536364 | - | accuracy=0.7553, logloss=0.5478 | ok |
| fastica | istella | algos | 1588 | 1588 | sklearn-cpu | 16796 | 0.09 | 0.09 |  | mean_abs_excess_kurtosis=356.3 | mean_abs_excess_kurtosis=356.3 | mean_abs_excess_kurtosis=922.5 | error |
| ard | istella | algos | 53561 | 983 | sklearn-cpu | 10421 | 5.14 | 0.09 | FLIP faster | finite=1, r2=-0.124856, rmse=0.885949 | - | r2=0.3274, rmse=0.6851 | ok |
| croston | taxi-hourly | algos | 2.9 | 4.9 | statsforecast-cpu | 54.2 | 0.05 | 0.09 |  | forecast_rmse=1.39026 | - | forecast_rmse=1.39 | ok |
| adaboost-clf | taxi | algos | 3037 | 2399 | sklearn-cpu | 27304 | 0.11 | 0.09 |  | accuracy=0.76523, logloss=0.543227 | - | accuracy=0.7654, logloss=0.5406 | ok |
| logreg | istella | classical2 | 3786 | 2533 | sklearn-cpu | 28852 | 0.13 | 0.09 |  | accuracy=0.92459, logloss=0.181249, nonfinite_proba_rows=0 | - | accuracy=0.9245, logloss=0.1813, nonfinite_proba_rows=0 | ok |
| affinity-prop | taxi | algos | 973 | 423 | sklearn-cpu | 4839 | 0.20 | 0.09 |  | n_clusters=272, silhouette=0.184644 | - | ari_vs_ours=1, n_clusters=272, silhouette=0.1846 | ok |
| logreg | taxi | classical2 | 130 | 26.7 | sklearn-cpu | 310 | 0.42 | 0.09 |  | accuracy=0.76335, logloss=0.538985, nonfinite_proba_rows=0 | - | accuracy=0.7633, logloss=0.539, nonfinite_proba_rows=0 | ok |
| bagging-clf | taxi | algos | 560 | 189 | sklearn-cpu | 2206 | 0.25 | 0.09 |  | accuracy=0.76749, logloss=0.531007 | - | accuracy=0.7677, logloss=0.5301 | ok |
| logreg-cv | istella | algos | - | 6204 | sklearn-cpu | 72840 | - | 0.09 |  | accuracy=0.9245, logloss=0.181393 | - | accuracy=0.9246, logloss=0.1814 | ok |
| ovr | taxi | algos | 376 | 113 | sklearn-cpu | 1461 | 0.26 | 0.08 |  | accuracy=0.47893 | - | accuracy=0.4789 | ok |
| binarizer | taxi | algos | 0.1 | 0.1 | sklearn-cpu | 1.6 | 0.07 | 0.08 |  | - | - | - | ok |
| bagging-reg | taxi | algos | 510 | 167 | sklearn-cpu | 2202 | 0.23 | 0.08 |  | finite=1, r2=0.916956, rmse=4.58978 | - | r2=0.9388, rmse=3.94 | ok |
| mds | istella | algos | 249 | 197 | sklearn-cpu | 2666 | 0.09 | 0.07 |  | trustworthiness_k15=0.586415 | - | trustworthiness_k15=0.5802 | ok |
| normalizer | taxi | algos | 0.1 | 0.1 | sklearn-cpu | 1.6 | 0.06 | 0.07 |  | - | - | - | ok |
| multioutput-clf | istella | algos | 2176 | 1631 | sklearn-cpu | 22340 | 0.10 | 0.07 |  | accuracy=0.959195 | - | accuracy=0.9592 | ok |
| mb-sparse-pca | taxi | algos | 187 | 161 | sklearn-cpu | 2223 | 0.08 | 0.07 |  | component_sparsity=0.0227273, relative_reconstruction_error=0.275935 | - | component_sparsity=0.02273, relative_reconstruction_error=0.2759 | ok |
| hdbscan | istella | classical | 51804 | 44988 | sklearn-cpu (fill) | 645384 | 0.08 | 0.07 |  | n_clusters=47, noise_fraction=0.25381, rows=100000 | - | n_clusters=47 | ok |
| adaboost-reg | istella | algos | 7517 | 12599 | sklearn-cpu | 183985 | 0.04 | 0.07 |  | finite=1, r2=0.229908, rmse=0.733047 | - | r2=0.1676, rmse=0.7621 | ok |
| croston-sba | taxi-hourly | algos | 2.5 | 3.2 | statsforecast-cpu | 47.9 | 0.05 | 0.07 |  | forecast_rmse=1.38668 | - | forecast_rmse=1.387 | ok |
| classical-mds | istella | algos | - | 176 | sklearn-cpu | 2623 | - | 0.07 |  | trustworthiness_k15=0.830548 | - | trustworthiness_k15=0.8306 | ok |
| mds | taxi | algos | 198 | 168 | sklearn-cpu | 2632 | 0.08 | 0.06 |  | trustworthiness_k15=0.606536 | - | trustworthiness_k15=0.6041 | ok |
| stacking-clf | istella | algos | 4997 | 9355 | sklearn-cpu | 149799 | 0.03 | 0.06 |  | accuracy=0.92997, logloss=0.193693 | - | accuracy=0.93, logloss=0.1939 | ok |
| classical-mds | taxi | algos | - | 157 | sklearn-cpu | 2535 | - | 0.06 |  | trustworthiness_k15=0.765649 | - | trustworthiness_k15=0.7657 | ok |
| voting-clf | taxi | algos | 455 | 83.4 | sklearn-cpu | 1597 | 0.29 | 0.05 |  | accuracy=0.74229, logloss=0.554797 | - | accuracy=0.7423, logloss=0.5546 | ok |
| ovr | istella | algos | 5172 | 3896 | sklearn-cpu | 75802 | 0.07 | 0.05 |  | accuracy=0.89271 | - | accuracy=0.8927 | ok |
| bpe-train | enwik8 | algos | 47.9 | 47.5 | hf-tokenizers-cpu | 1224 | 0.04 | 0.04 |  | jaccard_vs_ours=1, n_tokens=4096 | - | jaccard_vs_ours=0.9995, n_tokens=4096 | ok |
| croston | synthetic | algos | 2.6 | 3.1 | statsforecast-cpu | 91.7 | 0.03 | 0.03 |  | forecast_rmse=1.67484 | - | forecast_rmse=1.675 | ok |
| jl-min-dim | synthetic | algos | 5.5 | 5.4 | sklearn-cpu | 170 | 0.03 | 0.03 |  | - | - | equal_fraction_vs_sklearn=1 | ok |
| agglomerative | istella | classical2 | 151 | 152 | sklearn-cpu | 4827 | 0.03 | 0.03 |  | n_clusters=8, silhouette=0.716728 | - | ari_vs_ours=1, n_clusters=8, silhouette=0.7167 | ok |
| bagging-reg | istella | algos | 4216 | 1107 | sklearn-cpu | 36039 | 0.12 | 0.03 |  | finite=1, r2=0.523619, rmse=0.576551 | - | r2=0.5185, rmse=0.5797 | ok |
| bootstrap | istella | algos | 10.5 | 15.9 | scipy-cpu | 525 | 0.02 | 0.03 |  | ci_high=0.29485, ci_low=0.27125, standard_error=0.00598712 | - | ci_high=0.2955, ci_low=0.2717, standard_error=0.006044 | ok |
| bagging-clf | istella | algos | 4303 | 1134 | sklearn-cpu | 37699 | 0.11 | 0.03 |  | accuracy=0.9417, logloss=0.149948 | - | accuracy=0.9414, logloss=0.1521 | ok |
| decision-tree-clf | taxi | algos | 65.7 | 62.2 | sklearn-cpu | 2218 | 0.03 | 0.03 |  | accuracy=0.7563, logloss=1.20224 | - | accuracy=0.7566, logloss=1.15 | ok |
| decision-tree-reg | taxi | algos | 60.6 | 55.9 | sklearn-cpu | 2117 | 0.03 | 0.03 |  | finite=1, r2=0.862608, rmse=5.90362 | - | r2=0.8915, rmse=5.246 | ok |
| power-transformer | istella | algos | 433 | 7060 | sklearn-cpu | 276767 | 0.00 | 0.03 |  | - | - | - | ok |
| voting-clf | istella | algos | 1756 | 1192 | sklearn-cpu | 47440 | 0.04 | 0.03 |  | accuracy=0.91835, logloss=0.1877 | - | accuracy=0.9185, logloss=0.1875 | ok |
| radius-neighbors | istella | algos | 0.2 | 0.1 | sklearn-cpu | 5.7 | 0.03 | 0.02 |  | neighbors_total=1.22072e+06 | - | neighbors_total=1.221e+06 | ok |
| poisson | istella | algos | - | 2624 | sklearn-cpu (fill) | 110378 | - | 0.02 |  | finite=1, r2=0.243602, rmse=0.7265 | - | - | ok |
| bootstrap | taxi | algos | 14.1 | 12.3 | scipy-cpu | 525 | 0.03 | 0.02 |  | ci_high=18.7174, ci_low=18.2494, standard_error=0.117837 | - | ci_high=18.71, ci_low=18.25, standard_error=0.1176 | ok |
| select-mutual-info | istella | algos | 797 | 797 | sklearn-cpu | 36427 | 0.02 | 0.02 |  | n_selected=110 | n_selected=110 | jaccard_vs_sklearn=1, n_selected=110 | ok |
| kde | taxi | classical | 112 | 134 | sklearn-cpu | 6992 | 0.02 | 0.02 |  | mean_log_likelihood=-14.8265, rows_without_density=0 | - | mean_log_likelihood=-14.83, rows_without_density=0 | ok |
| permutation-test | istella | algos | - | 55.3 | scipy-cpu | 3042 | - | 0.02 |  | pvalue=0.1802, statistic=0.0112 | - | pvalue=0.1844, statistic=0.0112 | ok |
| permutation-test | taxi | algos | - | 55.3 | scipy-cpu | 3055 | - | 0.02 |  | pvalue=0.0006, statistic=-0.576702 | - | pvalue=0.001, statistic=-0.5767 | ok |
| adaboost-clf | istella | algos | 14883 | 11088 | sklearn-cpu (fill) | 615458 | 0.02 | 0.02 |  | accuracy=0.93499, logloss=0.438712 | - | accuracy=0.9371 | ok |
| gamma | istella | algos | - | 1889 | sklearn-cpu (fill) | 107798 | - | 0.02 |  | finite=1, r2=0.218164, rmse=0.738615 | - | - | ok |
| tweedie | istella | algos | - | 1886 | sklearn-cpu (fill) | 108690 | - | 0.02 |  | finite=1, r2=-24.4125, rmse=4.21099 | - | - | ok |
| huber | istella | algos | 46529 | 755 | sklearn-cpu | 45995 | 1.01 | 0.02 | FLIP faster | finite=1, r2=-0.00719273, rmse=0.838333 | - | r2=-0.01018, rmse=0.8396 | ok |
| optics | istella | algos | 534 | 450 | sklearn-cpu | 30143 | 0.02 | 0.01 |  | n_clusters=20, silhouette=-0.287356 | - | ari_vs_ours=0.9846, n_clusters=20, silhouette=-0.2858 | ok |
| decision-tree-reg | istella | algos | 476 | 438 | sklearn-cpu | 52136 | 0.01 | 0.01 |  | finite=1, r2=0.379438, rmse=0.658041 | - | r2=0.3732, rmse=0.6613 | ok |
| decision-tree-clf | istella | algos | 457 | 449 | sklearn-cpu | 54800 | 0.01 | 0.01 |  | accuracy=0.935, logloss=0.758303 | - | accuracy=0.9344, logloss=0.7917 | ok |
| normalizer | istella | algos | 0.1 | 0.1 | sklearn-cpu | 25.3 | 0.01 | 0.01 |  | - | - | - | ok |
| binarizer | istella | algos | 0.1 | 0.1 | sklearn-cpu | 25.3 | 0.01 | 0.01 |  | - | - | - | ok |
| meanshift | taxi | algos | 656 | 36.2 | sklearn-cpu | 9370 | 0.07 | 0.00 |  | n_clusters=122, silhouette=0.246631 | - | ari_vs_ours=1, n_clusters=122, silhouette=0.2466 | ok |
| kde | istella | classical | 181 | 178 | sklearn-cpu | 54651 | 0.00 | 0.00 |  | mean_log_likelihood=-222.271, rows_without_density=0 | - | mean_log_likelihood=-227, rows_without_density=0 | ok |
| radius-neighbors | taxi | algos | 0.2 | 0.1 | sklearn-cpu | 45.0 | 0.00 | 0.00 |  | neighbors_total=31 | - | neighbors_total=31 | ok |
| factor-analysis | taxi | algos | 157 | 362 | sklearn-cpu | 144607 | 0.00 | 0.00 |  | mean_log_likelihood=-14.8237 | - | mean_log_likelihood=-14.82 | ok |
| quantile | taxi | algos | 30309 | 496 | sklearn-cpu | 216369 | 0.14 | 0.00 |  | finite=1, r2=0.899596, rmse=5.04675 | - | r2=0.8997, rmse=5.045 | ok |
| linearsvc | istella | classical2 | 780 | 612 | sklearn-cpu (fill) | 593899 | 0.00 | 0.00 |  | accuracy=0.92347 | - | accuracy=0.9235 | ok |
| optics | taxi | algos | 417 | 425 | sklearn-cpu | 767657 | 0.00 | 0.00 |  | n_clusters=127, silhouette=-0.353359 | - | ari_vs_ours=1, n_clusters=127, silhouette=-0.3534 | ok |
| linearsvr | taxi | classical2 | 110 | 41.9 | sklearn-cpu | 83935 | 0.00 | 0.00 |  | finite=1, r2=0.899814, rmse=5.04127 | - | r2=0.8998, rmse=5.042 | ok |
| linearsvr | istella | classical2 | 973 | 186 | sklearn-cpu (fill) | 704105 | 0.00 | 0.00 |  | finite=1, r2=-0.106761, rmse=0.878794 | - | r2=-0.02573 | ok |
| adam | synthetic | algos | - | - | torch-eager-fp32 | 29.3 | - | - |  | - | - | - | MODE-MISMATCH(requested identical, read back unknown) |
| adamw | synthetic | algos | - | - | torch-eager-fp32 | 31.1 | - | - |  | - | - | - | MODE-MISMATCH(requested identical, read back unknown) |
| cnn-clf | synthetic | algos | 1106 | 1106 | - | - | - | - |  | accuracy=1 | accuracy=1 | - | ok |
| eigh | synthetic | algos | - | - | numpy-cpu | 4813 | - | - |  | - | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | REFUSED(timeout: null) |
| elliptic-envelope | istella | algos | - | - | sklearn-cpu | 43759 | - | - |  | - | - | fraction_flagged=0.09157, jaccard_vs_sklearn=1 | REFUSED(timeout: null) |
| embedding | synthetic | algos | - | - | torch-compile-fp32 | 11.1 | - | - |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | MODE-MISMATCH(requested identical, read back unknown) |
| gemm-int8 | gaussian | neural | 297 | 297 | - | - | - | - |  | max_rel_err_vs_fp64=0 | max_rel_err_vs_fp64=0 | - | ok |
| gmm | taxi | classical2 | 604 | 604 | - | - | - | - |  | bic=-3.67e+06, mean_log_likelihood=12.86, n_iter=32 | bic=-3.67e+06, mean_log_likelihood=12.86, n_iter=32 | - | ok |
| lamb | synthetic | algos | - | - | - | - | - | - |  | - | - | - | REFUSED(error: {"error": "Exception('lamb_step: offsets must rise strictly from 0, below 2^24')", "event": "error", "sta |
| lars | istella | algos | 4232 | 442 | - | - | - | - |  | finite=1, r2=0.309043, rmse=0.694362 | - | - | ok |
| lion | synthetic | algos | 135 | 135 | - | - | - | - |  | - | - | - | ok |
| min-cov-det | istella | algos | - | - | sklearn-cpu | 43632 | - | - |  | - | - | n_features=220 | REFUSED(timeout: null) |
| qn-reg | istella | algos | 427 | 371 | - | - | - | - |  | finite=1, r2=0.327491, rmse=0.68503 | - | - | ok |
| qn-reg | taxi | algos | 26.0 | 18.2 | - | - | - | - |  | finite=1, r2=0.908983, rmse=4.80504 | - | - | ok |
| quantile | istella | algos | - | 806 | sklearn-cpu: too slow to measure (216 s on taxi) | - | - | - |  | finite=1, r2=-0.0399981, rmse=0.851877 | - | - | ok |
| sgd | synthetic | algos | - | - | torch-eager-fp32 | 17.9 | - | - |  | - | - | - | MODE-MISMATCH(requested identical, read back unknown) |

Sources: before = M3 0.8.34 board (classical), M3 2026-09-29 board IDENTICAL cells (trees); job tags ident-adaboost-clf-istella, ident-adaboost-clf-taxi, ident-adaboost-reg-istella, ident-adaboost-reg-taxi, ident-additive-chi2-istella, ident-additive-chi2-taxi, ident-affinity-prop-istella, ident-affinity-prop-taxi, ident-agglomerative-istella, ident-agglomerative-taxi, ident-als-taxi-zones, ident-als-text, ident-ard-istella, ident-ard-taxi, ident-arima-synthetic, ident-auto-theta-synthetic, ident-auto-theta-taxi-hourly, ident-autoarima-synthetic, ident-autoarima-taxi-hourly, ident-bagging-clf-istella, ident-bagging-clf-taxi, ident-bagging-reg-istella, ident-bagging-reg-taxi, ident-bayesian-gmm-istella, ident-bayesian-gmm-taxi, ident-bayesian-ridge-istella, ident-bayesian-ridge-taxi, ident-bernoulli-nb-istella, ident-bernoulli-nb-taxi, ident-binarizer-istella, ident-binarizer-taxi, ident-bisecting-kmeans-istella, ident-bisecting-kmeans-taxi, ident-bootstrap-istella, ident-bootstrap-taxi, ident-bpe-encode-enwik8, ident-bpe-train-enwik8, ident-cagra-istella, ident-cagra-taxi, ident-calibrated-istella, ident-calibrated-taxi, ident-categorical-nb-istella, ident-categorical-nb-taxi, ident-cca-istella, ident-cca-taxi, ident-cholesky-synthetic, ident-classical-mds-istella, ident-classical-mds-taxi, ident-complement-nb-istella, ident-complement-nb-taxi, ident-complement-nb-text, ident-connected-components-istella, ident-connected-components-taxi, ident-cross-val-score-istella, ident-cross-val-score-taxi, ident-croston-optimized-synthetic, ident-croston-optimized-taxi-hourly, ident-croston-sba-synthetic, ident-croston-sba-taxi-hourly, ident-croston-synthetic, ident-croston-taxi-hourly, ident-damped-ets-synthetic, ident-damped-ets-taxi-hourly, ident-dart-istella, ident-dart-reg-istella, ident-dart-reg-taxi, ident-dart-taxi, ident-decision-tree-clf-istella, ident-decision-tree-clf-taxi, ident-decision-tree-reg-istella, ident-decision-tree-reg-taxi, ident-dict-learning-istella, ident-dict-learning-taxi, ident-dynamic-optimized-theta-synthetic, ident-dynamic-optimized-theta-taxi-hourly, ident-dynamic-theta-synthetic, ident-dynamic-theta-taxi-hourly, ident-elliptic-envelope-taxi, ident-enet-cv-istella, ident-enet-cv-taxi, ident-ets-synthetic, ident-factor-analysis-istella, ident-factor-analysis-taxi, ident-fastica-istella, ident-fastica-taxi, ident-gamma-istella, ident-gamma-taxi, ident-garch-synthetic, ident-garch-taxi-hourly, ident-gaussian-nb-istella, ident-gaussian-nb-taxi, ident-gaussian-rp-istella, ident-gaussian-rp-taxi, ident-gmm-istella, ident-gpr-istella, ident-gpr-taxi, ident-hdbscan-istella, ident-hdbscan-taxi, ident-huber-istella, ident-huber-taxi, ident-incremental-pca-istella, ident-incremental-pca-taxi, ident-isomap-istella, ident-isomap-taxi, ident-isotonic-istella, ident-isotonic-taxi, ident-iterative-imputer-istella, ident-iterative-imputer-taxi, ident-ivf-filter-istella, ident-ivf-filter-taxi, ident-ivf-istella, ident-ivf-pq-istella, ident-ivf-pq-taxi, ident-ivf-rabitq-istella, ident-ivf-rabitq-taxi, ident-ivf-refine-istella, ident-ivf-sq-istella, ident-ivf-sq-taxi, ident-ivf-taxi, ident-jl-min-dim-synthetic, ident-kbins-istella, ident-kbins-taxi, ident-kde-istella, ident-kde-taxi, ident-kernel-pca-istella, ident-kernel-pca-taxi, ident-kernel-ridge-istella, ident-kernel-ridge-taxi, ident-kernel-shap-istella, ident-kernel-shap-taxi, ident-kmeans-istella, ident-kmeans-taxi, ident-knn-clf-istella, ident-knn-clf-taxi, ident-knn-imputer-istella, ident-knn-imputer-taxi, ident-knn-istella, ident-knn-reg-istella, ident-knn-reg-taxi, ident-knn-taxi, ident-kpss-synthetic, ident-kpss-taxi-hourly, ident-label-binarizer-istella, ident-label-binarizer-taxi, ident-label-encoder-istella, ident-label-encoder-taxi, ident-label-propagation-istella, ident-label-propagation-taxi, ident-label-spreading-istella, ident-label-spreading-taxi, ident-lars-istella, ident-lars-taxi, ident-lasso-cv-istella, ident-lasso-cv-taxi, ident-lasso-lars-istella, ident-lasso-lars-taxi, ident-lda-clf-istella, ident-lda-clf-taxi, ident-lda-taxi-zones, ident-lda-text, ident-linearsvc-istella, ident-linearsvc-taxi, ident-linearsvr-istella, ident-linearsvr-taxi, ident-lle-istella, ident-lle-taxi, ident-lof-istella, ident-lof-taxi, ident-logreg-cv-istella, ident-logreg-cv-taxi, ident-logreg-istella, ident-logreg-taxi, ident-louvain-istella, ident-louvain-taxi, ident-lstsq-istella, ident-lstsq-taxi, ident-lu-factor-synthetic, ident-lu-solve-synthetic, ident-maxabs-scaler-istella, ident-maxabs-scaler-taxi, ident-mb-dict-learning-istella, ident-mb-dict-learning-taxi, ident-mb-sparse-pca-istella, ident-mb-sparse-pca-taxi, ident-mds-istella, ident-mds-taxi, ident-meanshift-istella, ident-meanshift-taxi, ident-min-cov-det-taxi, ident-minibatch-kmeans-istella, ident-minibatch-kmeans-taxi, ident-minmax-scaler-istella, ident-minmax-scaler-taxi, ident-multilabel-binarizer-istella, ident-multilabel-binarizer-taxi, ident-multinomial-nb-istella, ident-multinomial-nb-taxi, ident-multinomial-nb-text, ident-multioutput-clf-istella, ident-multioutput-clf-taxi, ident-multioutput-reg-istella, ident-multioutput-reg-taxi, ident-nearest-centroid-istella, ident-nearest-centroid-taxi, ident-nmf-istella, ident-nmf-taxi, ident-normalizer-istella, ident-normalizer-taxi, ident-nystroem-istella, ident-nystroem-taxi, ident-ocsvm-istella, ident-ocsvm-taxi, ident-ols-istella, ident-ols-taxi, ident-onehot-istella, ident-onehot-taxi, ident-optics-istella, ident-optics-taxi, ident-optimized-theta-synthetic, ident-optimized-theta-taxi-hourly, ident-ordinal-istella, ident-ordinal-taxi, ident-ovr-istella, ident-ovr-taxi, ident-pa-clf-istella, ident-pa-clf-taxi, ident-pa-reg-istella, ident-pa-reg-taxi, ident-pagerank-istella, ident-pagerank-taxi, ident-pca-istella, ident-pca-taxi, ident-perceptron-istella, ident-perceptron-taxi, ident-permutation-shap-istella, ident-permutation-shap-taxi, ident-permutation-test-istella, ident-permutation-test-taxi, ident-pls-canonical-istella, ident-pls-canonical-taxi, ident-pls-istella, ident-pls-taxi, ident-poisson-istella, ident-poisson-taxi, ident-poly-count-sketch-istella, ident-poly-count-sketch-taxi, ident-poly-features-istella, ident-poly-features-taxi, ident-power-transformer-istella, ident-power-transformer-taxi, ident-prophet-synthetic, ident-prophet-taxi-hourly, ident-qda-istella, ident-qda-taxi, ident-qn-reg-istella, ident-qn-reg-taxi, ident-qr-istella, ident-qr-taxi, ident-quantile-istella, ident-quantile-taxi, ident-quantile-transformer-istella, ident-quantile-transformer-taxi, ident-radius-neighbors-istella, ident-radius-neighbors-taxi, ident-random-trees-embedding-istella, ident-random-trees-embedding-taxi, ident-randomized-svd-istella, ident-randomized-svd-taxi, ident-rbf-sampler-istella, ident-rbf-sampler-taxi, ident-resample-istella, ident-resample-taxi, ident-rfe-istella, ident-rfe-taxi, ident-ridge-clf-istella, ident-ridge-clf-taxi, ident-ridge-cv-istella, ident-ridge-cv-taxi, ident-ridge-istella, ident-ridge-taxi, ident-robust-scaler-istella, ident-robust-scaler-taxi, ident-select-chi2-istella, ident-select-chi2-taxi, ident-select-d-synthetic, ident-select-d-taxi-hourly, ident-select-f-classif-istella, ident-select-f-classif-taxi, ident-select-f-regression-istella, ident-select-f-regression-taxi, ident-select-mutual-info-reg-taxi, ident-select-mutual-info-taxi, ident-select-r-regression-istella, ident-select-r-regression-taxi, ident-sgd-ocsvm-istella, ident-sgd-ocsvm-taxi, ident-simple-imputer-istella, ident-simple-imputer-taxi, ident-skewed-chi2-istella, ident-skewed-chi2-taxi, ident-sparse-coder-istella, ident-sparse-coder-taxi, ident-sparse-pca-istella, ident-sparse-pca-taxi, ident-sparse-rp-istella, ident-sparse-rp-taxi, ident-spectral-embedding-istella, ident-spectral-embedding-taxi, ident-spectral-istella, ident-spectral-taxi, ident-spline-istella, ident-spline-taxi, ident-stacking-clf-istella, ident-stacking-clf-taxi, ident-stacking-reg-istella, ident-stacking-reg-taxi, ident-standard-scaler-istella, ident-standard-scaler-taxi, ident-stl-synthetic, ident-stl-taxi-hourly, ident-svc-istella, ident-svc-taxi, ident-svd-istella, ident-svd-taxi, ident-svgp-istella, ident-svgp-taxi, ident-svr-istella, ident-svr-taxi, ident-target-encoder-istella, ident-target-encoder-taxi, ident-theta-synthetic, ident-theta-taxi-hourly, ident-tree-shap-istella, ident-tree-shap-taxi, ident-tsne-istella, ident-tsne-taxi, ident-tsvd-istella, ident-tsvd-taxi, ident-tweedie-istella, ident-tweedie-taxi, ident-umap-istella, ident-umap-taxi, ident-var-synthetic, ident-var-taxi-hourly, ident-variance-threshold-istella, ident-variance-threshold-taxi, ident-voting-clf-istella, ident-voting-clf-taxi, ident-voting-reg-istella, ident-voting-reg-taxi.
