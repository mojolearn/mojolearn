# M3 IDENTICAL board

Generated from board.json by tools/af_board_render.py; do not edit. Source: `bench/results/bench_board/m3ultra-0834/board.json` (the one M3 board; BOARD.md, BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md are rendered from it).

## Checks

- IDENTICAL cells come from M3 ident sweeps (`tools/af_board_ident_update.py`, which writes board.json and re-renders every page); same-bits checks are separate (`lq add <box> ID ...`).
- Render check: `python3 tools/af_board_render.py --check` fails when BOARD.md, BOARD_M3_FAST.md or BOARD_M3_IDENTICAL.md differ from a fresh render of board.json. `tools/af_board_apply.py` and `tools/af_board_ident_update.py` always run it; `tools/af_board_render_watch.sh` (called from ~/mojolearn-evidence/apple_watch.sh) runs it against origin/main and prints ALERT on a mismatch.

Our IDENTICAL arm on the M3 Ultra Metal GPU at head cf94a6be6, 1 unscored warm-up + 1 scored round at board size (the 2026-10-04 ident-* sweep, 350 jobs, AFC_ARM=ours) (rows-full; trees MOJOLEARN_SPEED_SIZE=shipped). Opponents are not re-raced: classical times come from the M3 0.8.34 board (`~/mojolearn-evidence/board-0834-times.tsv`, quality from its board.json), trees from the 2026-09-29 M3 board (older tree params on some lanes); an opponent marked (fill) comes from the M3 opponent fill on the 0.8.34 kit. Ratio = our IDENTICAL ms / best opponent ms; below 1 is faster. Rows sort worst ratio after first. First written by `tools/af_board_merge.py`; now carried in board.json.

Quality (2026-10-04, computed): of 465 lane/dataset rows, IDENTICAL quality is WORSE than an opponent on 69 (49 by more than 1%) and IDENTICAL after is WORSE than IDENTICAL before on 0 (0 by more than 1%); 2 rows carry metrics of unknown direction (1 metrics), 117 have no parseable IDENTICAL quality. Tolerance rel 1e-3, abs 1e-6.

Canonical full-board summary (2026-10-04): 421 IDENTICAL rows, 414 eligible opponent comparisons, 272 faster, geometric-mean ratio 0.482. Eligible = a ratio after and a status without HOLD or "excluded"; faster = ratio below 1. Ratio = IDENTICAL ms / best opponent ms.

- Hand-made headline, last before the page was generated: Summary: 350 rows, 329 with a ratio, 203 faster than the best opponent after, geometric-mean ratio 0.48. Flips to faster: ard taxi, svd taxi, select-r-regression istella, gpr taxi, svd istella, enet-cv taxi, bisecting-kmeans taxi, connected-components istella, label-encoder istella, perceptron istella, multinomial-nb istella, complement-nb istella, lasso-cv taxi, minmax-scaler taxi, maxabs-scaler taxi, robust-scaler taxi, tree-shap istella, nearest-centroid istella, meanshift istella, isotonic taxi, pa-clf istella, ridge-clf istella, dict-learning istella, mb-sparse-pca istella, complement-nb taxi, multinomial-nb taxi, label-encoder taxi, gaussian-nb istella, lasso-cv istella, qr taxi, pa-reg istella, tree-shap taxi, enet-cv istella, lstsq istella, gaussian-nb taxi, categorical-nb taxi, categorical-nb istella, bernoulli-nb taxi, lda taxi-zones, huber taxi, bayesian-ridge istella, ard istella, huber istella. Flips to slower: pagerank taxi, pagerank istella, umap istella, rbf-sampler taxi, ridge taxi, permutation-shap taxi, kernel-shap taxi, louvain istella, spectral taxi, ols taxi, ivf istella, select-f-regression istella.

| lane | dataset | family | IDENTICAL before ms | IDENTICAL after ms | best opponent | opp ms | ratio before | ratio after | flip | quality after (IDENTICAL) | quality before (IDENTICAL) | opponent quality | status |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| elliptic-envelope | taxi | algos | 79473 | 1292934 | sklearn-cpu | 1423 | 55.84 | 908.38 |  | fraction_flagged=0.10237 | - | fraction_flagged=0.1025, jaccard_vs_sklearn=1 | ok |
| min-cov-det | taxi | algos | 80018 | 1265140 | sklearn-cpu | 1413 | 56.61 | 895.08 |  | n_features=11 | - | n_features=11 | ok |
| cagra | istella | algos | 181960 | 182595 | faiss-cpu | 1634 | 111.39 | 111.78 |  | recall_at_10=0.9838 | - | recall_at_10=0.9994 | ok |
| cross-entropy | synthetic | algos | 324 | 324 | torch-compile-fp32 | 3.6 | 89.33 | 89.33 |  | loss_rel_err_vs_fp64=4.564e-08 | loss_rel_err_vs_fp64=4.564e-08 | grad_max_rel_diff_vs_ours=5.588e-09, loss_rel_err_vs_fp64=4.564e-08 | ok |
| avgpool2d | synthetic | algos | 146 | 146 | torch-compile-bf16 | 2.5 | 57.32 | 57.32 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| sparse-coder | istella | algos | 0.2 | 1.4 | sklearn-cpu | 0.0 | 7.70 | 47.84 |  | - | - | - | ok |
| resnet-block | synthetic | algos | 486 | 486 | torch-compile-bf16 | 10.5 | 46.44 | 46.44 |  | - | - | max_rel_diff_vs_torch_eager_fp32=1.85e+04, rel_fro_vs_torch_eager_fp32=0.003425 | ok |
| sparse-coder | taxi | algos | 0.2 | 1.7 | sklearn-cpu | 0.0 | 4.74 | 40.23 |  | - | - | - | ok |
| graphsage | istella | algos | 387 | 387 | torch-compile-bf16 | 10.2 | 38.02 | 38.02 |  | - | - | max_rel_diff_vs_torch_eager_fp32=3678, rel_fro_vs_torch_eager_fp32=0.003054 | ok |
| maxpool2d | synthetic | algos | 156 | 156 | torch-eager-bf16 | 4.2 | 37.34 | 37.34 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| dropout2d | synthetic | algos | 82.7 | 82.7 | torch-compile-fp32 | 2.2 | 37.05 | 37.05 |  | - | - | - | ok |
| maxpool1d | synthetic | algos | 63.1 | 63.1 | torch-compile-bf16 | 2.0 | 30.93 | 30.93 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| batchnorm1d | synthetic | algos | 72.6 | 72.6 | torch-eager-bf16 | 2.4 | 30.42 | 30.42 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| conv1d | synthetic | algos | 237 | 237 | torch-compile-bf16 | 8.0 | 29.67 | 29.67 |  | - | - | max_rel_diff_vs_torch_eager_fp32=3418, rel_fro_vs_torch_eager_fp32=0.003296 | ok |
| avgpool1d | synthetic | algos | 50.0 | 50.0 | torch-compile-fp32 | 1.7 | 29.40 | 29.40 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| batchnorm2d | synthetic | algos | 62.1 | 62.1 | torch-eager-bf16 | 2.3 | 26.71 | 26.71 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| moe | synthetic | algos | 820 | 820 | torch-eager-bf16 | 37.7 | 21.74 | 21.74 |  | - | - | max_rel_diff_vs_torch_eager_fp32=2.234e+04, rel_fro_vs_torch_eager_fp32=0.05522 | ok |
| conv2d | synthetic | algos | 109 | 109 | torch-compile-bf16 | 5.4 | 20.18 | 20.18 |  | - | - | max_rel_diff_vs_torch_eager_fp32=3418, rel_fro_vs_torch_eager_fp32=0.003383 | ok |
| graphsage | taxi | algos | 123 | 123 | torch-compile-bf16 | 6.2 | 20.04 | 20.04 |  | - | - | max_rel_diff_vs_torch_eager_fp32=4608, rel_fro_vs_torch_eager_fp32=0.003285 | ok |
| lr-onecycle | synthetic | algos | 1385 | 1385 | torch-cpu | 111 | 12.50 | 12.50 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.951e-08 | ok |
| gcn | istella | algos | 194 | 194 | torch-compile-fp32 | 18.8 | 10.34 | 10.34 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0.01759, rel_fro_vs_torch_eager_fp32=1.004e-07 | ok |
| layernorm | synthetic | algos | 27.6 | 27.6 | torch-eager-bf16 | 2.8 | 10.02 | 10.02 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| adagrad | synthetic | algos | 131 | 131 | torch-eager-fp32 | 14.2 | 9.22 | 9.22 |  | - | - | - | ok |
| gcn | taxi | algos | 144 | 144 | torch-compile-fp32 | 17.8 | 8.13 | 8.13 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0.003725, rel_fro_vs_torch_eager_fp32=9.327e-08 | ok |
| lm-forward | bytes | neural | 93.3 | 93.3 | torch-compile-bf16 | 11.9 | 7.83 | 7.83 |  | mean_nll=9.019 | mean_nll=9.019 | mean_nll=9.019 | ok |
| lr-warmup-linear | synthetic | algos | 780 | 780 | torch-cpu | 101 | 7.68 | 7.68 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0.001 | ok |
| autoarima | taxi-hourly | algos | 57797 | 19386 | statsforecast-cpu | 2844 | 20.32 | 6.82 |  | forecast_rmse=73.6303 | - | forecast_rmse=68.21 | ok |
| svgp | taxi | algos | 1040 | 504 | gpytorch-gpu | 75.4 | 13.79 | 6.68 |  | finite=1, r2=-0.194982, rmse=17.7235 | - | r2=-0.2094, rmse=17.83 | ok |
| lr-constant | synthetic | algos | 570 | 570 | torch-cpu | 87.4 | 6.52 | 6.52 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=1.038e-07 | ok |
| gemm | gaussian | neural | 72.3 | 72.3 | torch-eager-bf16 | 11.1 | 6.49 | 6.49 |  | max_rel_err_vs_fp64=2.399e-07 | max_rel_err_vs_fp64=2.399e-07 | max_rel_err_vs_fp64=0.003752 | ok |
| adamax | synthetic | algos | 176 | 176 | torch-eager-fp32 | 27.7 | 6.35 | 6.35 |  | - | - | - | ok |
| autoarima | synthetic | algos | 39660 | 17968 | statsforecast-cpu | 2880 | 13.77 | 6.24 |  | forecast_rmse=3.41284 | - | forecast_rmse=17.55 | ok |
| svgp | istella | algos | 1094 | 522 | gpytorch-gpu | 84.9 | 12.88 | 6.15 |  | finite=1, r2=-0.106016, rmse=0.878373 | - | r2=-0.106, rmse=0.8784 | ok |
| damped-ets | synthetic | algos | 1017 | 1017 | statsforecast-cpu | 167 | 6.08 | 6.08 |  | forecast_rmse=13.9312 | - | forecast_rmse=13.95 | ok |
| damped-ets | taxi-hourly | algos | 1121 | 1122 | statsforecast-cpu | 185 | 6.06 | 6.06 |  | forecast_rmse=96.6904 | - | forecast_rmse=96.69 | ok |
| nadam | synthetic | algos | 180 | 180 | torch-eager-fp32 | 32.9 | 5.46 | 5.46 |  | - | - | - | ok |
| rmsprop | synthetic | algos | 131 | 131 | torch-eager-fp32 | 25.2 | 5.21 | 5.21 |  | - | - | - | ok |
| lr-warmup-cosine | synthetic | algos | 539 | 539 | torch-cpu | 114 | 4.74 | 4.74 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | - | ok |
| adafactor | synthetic | algos | 1588 | 1588 | torch-eager-fp32 | 343 | 4.62 | 4.62 |  | - | - | - | ok |
| label-binarizer | taxi | algos | 559 | 540 | sklearn-cpu | 118 | 4.75 | 4.59 |  | - | - | - | ok |
| ridge-cv | taxi | algos | 4306 | 4318 | sklearn-cpu | 946 | 4.55 | 4.56 |  | finite=1, r2=0.908981, rmse=4.80511 | - | r2=0.909, rmse=4.805 | ok |
| additive-chi2 | taxi | algos | 0.7 | 1.8 | sklearn-cpu | 0.4 | 1.79 | 4.41 |  | kernel_rel_error=0.0938923 | - | kernel_rel_error=0.09389 | ok |
| lm-train-step | bytes | neural | 185 | 185 | torch-compile-bf16 | 43.0 | 4.30 | 4.30 |  | loss_first_step=9.019, loss_last_step=8.418, steps=2 | loss_first_step=9.019, loss_last_step=8.418, steps=2 | loss_first_step=9.018, loss_last_step=8.416, steps=2 | ok |
| gemm-bf16 | gaussian | neural | 69.1 | 69.1 | torch-eager-bf16 | 16.2 | 4.27 | 4.27 |  | max_rel_err_vs_fp64=1.155e-07 | max_rel_err_vs_fp64=1.155e-07 | max_rel_err_vs_fp64=0.002759 | ok |
| stl | synthetic | algos | 414 | 414 | statsmodels-cpu | 105 | 3.95 | 3.95 |  | residual_std=0.783175 | - | residual_std=0.7832 | ok |
| cnn-clf | synthetic | algos | 1106 | 1106 | torch-compile-bf16 | 286 | 3.86 | 3.86 |  | accuracy=1 | accuracy=1 | accuracy=1 | ok |
| onehot | taxi | algos | 89.5 | 68.6 | sklearn-cpu | 19.0 | 4.71 | 3.61 |  | - | - | - | ok |
| sgd-ocsvm | taxi | algos | 91217 | 3232 | sklearn-cpu | 898 | 101.57 | 3.60 |  | fraction_flagged=0.28379 | - | fraction_flagged=0.00702, jaccard_vs_sklearn=1 | ok |
| ordinal | taxi | algos | 83.9 | 67.5 | sklearn-cpu | 18.8 | 4.47 | 3.60 |  | - | - | - | ok |
| pca | istella | classical | 819 | 725 | sklearn-cpu | 205 | 3.99 | 3.53 |  | explained_variance_ratio_sum=1 | - | explained_variance_ratio_sum=1 | ok |
| stl | taxi-hourly | algos | 415 | 414 | statsmodels-cpu | 119 | 3.49 | 3.49 |  | residual_std=18.3125 | - | residual_std=18.31 | ok |
| global-avgpool | synthetic | algos | 4.1 | 4.1 | torch-compile-bf16 | 1.2 | 3.48 | 3.48 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0.007299, rel_fro_vs_torch_eager_fp32=9.321e-08 | ok |
| pagerank | taxi | algos | 226 | 226 | networkx-cpu | 64.9 | 3.48 | 3.48 |  | sum=1 | - | sum=1 | ok |
| cagra | taxi | algos | 3532 | 3507 | faiss-cpu | 1016 | 3.48 | 3.45 |  | recall_at_10=0.4838 | - | recall_at_10=0.9277 | ok |
| minibatch-kmeans | istella | algos | 405 | 412 | sklearn-cpu | 123 | 3.28 | 3.34 |  | n_clusters=8, silhouette=0.116696 | - | ari_vs_ours=0.6224, n_clusters=8, silhouette=0.1119 | ok |
| calibrated | taxi | algos | 1422 | 1839 | sklearn-cpu | 556 | 2.56 | 3.31 |  | accuracy=0.75533, logloss=0.550718 | - | accuracy=0.7553, logloss=0.5507 | ok |
| samba-train-step | bytes | neural | 339 | 339 | torch-eager-fp32 | 103 | 3.28 | 3.28 |  | loss_first_step=5.636, loss_last_step=4.834, steps=2 | loss_first_step=5.636, loss_last_step=4.834, steps=2 | loss_first_step=5.636, loss_last_step=4.834, steps=2 | ok |
| global-maxpool | synthetic | algos | 4.3 | 4.3 | torch-eager-bf16 | 1.3 | 3.27 | 3.27 |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | ok |
| kernel-shap | istella | algos | 27059 | 24469 | shap-cpu | 7695 | 3.52 | 3.18 |  | rel_error_vs_exact=4.37806e-09 | - | rel_error_vs_exact=1.405e-14 | ok |
| additive-chi2 | istella | algos | 11.0 | 11.3 | sklearn-cpu | 3.8 | 2.89 | 2.96 |  | kernel_rel_error=0.0877304 | - | kernel_rel_error=0.08773 | ok |
| lr-exponential | synthetic | algos | 225 | 225 | torch-cpu | 76.7 | 2.94 | 2.94 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=5.933e-08 | ok |
| lstm-reg | synthetic | algos | 1984 | 1984 | torch-compile-fp32 | 696 | 2.85 | 2.85 |  | r2=0.981, rmse=0.1596 | r2=0.981, rmse=0.1596 | r2=0.981, rmse=0.1596 | ok |
| lstm-reg | taxi-hourly | algos | 1989 | 1989 | torch-compile-fp32 | 698 | 2.85 | 2.85 |  | r2=0.7517, rmse=0.5404 | r2=0.7517, rmse=0.5404 | r2=0.7517, rmse=0.5404 | ok |
| transformer-forward | gaussian | neural | 14.1 | 14.1 | torch-compile-fp32 | 5.0 | 2.82 | 2.82 |  | - | - | - | ok |
| lstm-clf | synthetic | algos | 1991 | 1991 | torch-eager-fp32 | 727 | 2.74 | 2.74 |  | accuracy=0.9687, logloss=0.07244 | accuracy=0.9687, logloss=0.07244 | accuracy=0.9687 | ok |
| lstm-clf | taxi-hourly | algos | 1991 | 1991 | torch-compile-fp32 | 733 | 2.72 | 2.72 |  | accuracy=0.8682, logloss=0.2999 | accuracy=0.8682, logloss=0.2999 | accuracy=0.8682 | ok |
| pagerank | istella | algos | 224 | 221 | networkx-cpu | 82.0 | 2.73 | 2.69 |  | sum=1 | - | sum=1 | ok |
| minmax-scaler | istella | algos | 143 | 156 | sklearn-cpu | 63.0 | 2.27 | 2.47 |  | - | - | - | ok |
| spline | istella | algos | 166 | 43.1 | sklearn-cpu | 17.6 | 9.45 | 2.45 |  | - | - | - | ok |
| clip-grad-norm | synthetic | algos | 40.5 | 40.5 | torch-compile-fp32 | 17.0 | 2.37 | 2.37 |  | norm=1, norm_rel_diff_vs_ours=0 | norm=1, norm_rel_diff_vs_ours=0 | norm=1, norm_rel_diff_vs_ours=0 | ok |
| dynamic-optimized-theta | taxi-hourly | algos | 3196 | 3196 | statsforecast-cpu | 1352 | 2.36 | 2.36 |  | forecast_rmse=49.0866 | - | forecast_rmse=49.31 | ok |
| lars | taxi | algos | 82.1 | 80.0 | sklearn-cpu | 34.2 | 2.40 | 2.34 |  | finite=1, r2=0.908981, rmse=4.80511 | - | r2=0.909, rmse=4.805 | ok |
| lasso-lars | taxi | algos | 82.5 | 80.0 | sklearn-cpu | 34.6 | 2.38 | 2.31 |  | finite=1, r2=0.908996, rmse=4.8047 | - | r2=0.909, rmse=4.804 | ok |
| permutation-shap | istella | algos | 34179 | 28438 | shap-cpu | 12335 | 2.77 | 2.31 |  | rel_error_vs_exact=5.33875e-09 | - | rel_error_vs_exact=3.692e-10 | ok |
| calibrated | istella | algos | 5126 | 6363 | sklearn-cpu | 2784 | 1.84 | 2.29 |  | accuracy=0.88509, logloss=0.289832 | - | accuracy=0.8851, logloss=0.2898 | ok |
| cholesky | synthetic | algos | 445 | 251 | torch-gpu | 110 | 4.04 | 2.27 |  | relative_residual=2.90007e-07 | - | relative_residual=5.449e-07 | ok |
| lu-factor | synthetic | algos | 1429 | 969 | scipy-cpu | 428 | 3.34 | 2.26 |  | relative_residual=3.2563e-06 | - | relative_residual=3.246e-06 | ok |
| theta | taxi-hourly | algos | 385 | 385 | statsmodels-cpu | 171 | 2.25 | 2.25 |  | forecast_rmse=49.0206 | - | forecast_rmse=49.31 | ok |
| var | taxi-hourly | algos | 5.2 | 6.2 | statsmodels-cpu | 2.8 | 1.88 | 2.23 |  | forecast_rmse=33.168 | - | forecast_rmse=33.17 | ok |
| poly-count-sketch | taxi | algos | 0.3 | 1.1 | sklearn-cpu | 0.5 | 0.54 | 2.21 | FLIP slower | kernel_rel_error=0.0965957 | - | kernel_rel_error=0.0966 | ok |
| lasso | istella | classical2 | 8496 | 8496 | sklearn-cpu | 3879 | 2.19 | 2.19 |  | r2=0.3108, rmse=0.6935 | r2=0.3108, rmse=0.6935 | r2=0.3108, rmse=0.6935 | ok |
| onehot | istella | algos | 111 | 79.6 | sklearn-cpu | 36.7 | 3.03 | 2.17 |  | - | - | - | ok |
| ordinal | istella | algos | 111 | 79.7 | sklearn-cpu | 36.9 | 3.01 | 2.16 |  | - | - | - | ok |
| multilabel-binarizer | taxi | algos | 283 | 274 | sklearn-cpu | 129 | 2.19 | 2.11 |  | - | - | - | ok |
| skewed-chi2 | taxi | algos | 1.7 | 1.2 | sklearn-cpu | 0.6 | 3.07 | 2.09 |  | kernel_rel_error=0.0377486 | - | kernel_rel_error=0.03775 | ok |
| mlp-train-step | gaussian | neural | 8.4 | 8.4 | torch-compile-bf16 | 4.2 | 2.00 | 2.00 |  | loss_first_step=1.16, loss_last_step=1.123, steps=2 | loss_first_step=1.16, loss_last_step=1.123, steps=2 | loss_first_step=1.161, loss_last_step=1.123, steps=2 | ok |
| sparse-rp | taxi | algos | 3.9 | 3.5 | sklearn-cpu | 1.8 | 2.19 | 1.97 |  | mean_abs_distortion=0.147163 | - | mean_abs_distortion=0.381 | ok |
| kernel-pca | taxi | algos | 949 | 899 | sklearn-cpu | 461 | 2.06 | 1.95 |  | - | - | subspace_cos_vs_sklearn=1 | ok |
| randomized-svd | istella | algos | 718 | 692 | sklearn-cpu | 369 | 1.95 | 1.87 |  | relative_reconstruction_error=0.000235946 | - | relative_reconstruction_error=0.0002359 | ok |
| multioutput-reg | taxi | algos | 155 | 99.3 | sklearn-cpu | 53.7 | 2.90 | 1.85 |  | r2=0.60424 | - | r2=0.6043 | ok |
| bayesian-ridge | taxi | algos | 131 | 130 | sklearn-cpu | 73.2 | 1.79 | 1.78 |  | finite=1, r2=0.908979, rmse=4.80515 | - | r2=0.909, rmse=4.805 | ok |
| var | synthetic | algos | 5.3 | 4.7 | statsmodels-cpu | 2.6 | 1.99 | 1.77 |  | forecast_rmse=1.14086 | - | forecast_rmse=1.145 | ok |
| kernel-pca | istella | algos | 1036 | 981 | sklearn-cpu | 560 | 1.85 | 1.75 |  | - | - | subspace_cos_vs_sklearn=1 | ok |
| kpss | taxi-hourly | algos | 3.8 | 4.5 | statsmodels-cpu | 2.6 | 1.48 | 1.75 |  | stationary_fraction=0.6875 | - | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| elasticnet | taxi | classical2 | 68.5 | 68.5 | sklearn-cpu | 40.3 | 1.70 | 1.70 |  | r2=0.9074, rmse=4.847 | r2=0.9074, rmse=4.847 | r2=0.9074, rmse=4.847 | ok |
| knn | taxi | classical | 699 | 715 | sklearn-cpu | 426 | 1.64 | 1.68 |  | recall_at_k=0.999754, rows_with_repeated_ids=0 | - | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| permutation-shap | taxi | algos | 122 | 138 | shap-cpu | 83.0 | 1.46 | 1.66 |  | rel_error_vs_exact=2.1497e-08 | - | rel_error_vs_exact=1.279e-15 | ok |
| elasticnet | istella | classical2 | 2949 | 2949 | sklearn-cpu | 1819 | 1.62 | 1.62 |  | r2=0.2609, rmse=0.7181 | r2=0.2609, rmse=0.7181 | r2=0.2609, rmse=0.7181 | ok |
| spline | taxi | algos | 147 | 28.6 | sklearn-cpu | 17.6 | 8.31 | 1.62 |  | - | - | - | ok |
| lasso | taxi | classical2 | 76.1 | 76.1 | sklearn-cpu | 47.9 | 1.59 | 1.59 |  | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | r2=0.909, rmse=4.805 | ok |
| multinomial-nb | istella | algos | 136 | 287 | sklearn-cpu | 183 | 0.74 | 1.56 | FLIP slower | accuracy=0.85362, logloss=3.62857 | - | accuracy=0.8536, logloss=3.087 | ok |
| rbf-sampler | istella | classical2 | 101 | 72.4 | sklearn-cpu | 47.6 | 2.13 | 1.52 |  | kernel_rel_error=0.14198 | - | kernel_rel_error=0.1374 | ok |
| kpss | synthetic | algos | 3.7 | 4.0 | statsmodels-cpu | 2.6 | 1.41 | 1.51 |  | stationary_fraction=0.03125 | - | flag_agreement_vs_statsmodels=1, stat_max_rel_diff_vs_statsmodels=0, stationary_fraction=0 | ok |
| rbf-sampler | taxi | classical2 | 63.7 | 62.7 | sklearn-cpu | 42.2 | 1.51 | 1.49 |  | kernel_rel_error=0.108549 | - | kernel_rel_error=0.08378 | ok |
| minibatch-kmeans | taxi | algos | 87.0 | 65.1 | sklearn-cpu | 43.9 | 1.98 | 1.48 |  | n_clusters=8, silhouette=0.13806 | - | ari_vs_ours=0.5252, n_clusters=8, silhouette=0.1655 | ok |
| complement-nb | istella | algos | 135 | 269 | sklearn-cpu | 186 | 0.72 | 1.44 | FLIP slower | accuracy=0.84936, logloss=3.76252 | - | accuracy=0.8494, logloss=3.175 | ok |
| isotonic | istella | algos | 55.2 | 52.7 | sklearn-cpu | 37.5 | 1.47 | 1.41 |  | finite=1, r2=0.187985, rmse=0.752735 | - | r2=0.188, rmse=0.7527 | ok |
| mb-dict-learning | istella | algos | 7069 | 7040 | sklearn-cpu | 5030 | 1.41 | 1.40 |  | component_sparsity=0.0863636, relative_reconstruction_error=0.64834 | - | component_sparsity=0.08636, relative_reconstruction_error=0.644 | ok |
| select-mutual-info-reg | istella | algos | 69680 | 69680 | sklearn-cpu | 50021 | 1.39 | 1.39 |  | n_selected=110 | n_selected=110 | jaccard_vs_sklearn=1, n_selected=110 | ok |
| ridge | taxi | classical2 | 53.3 | 49.7 | sklearn-cpu | 36.1 | 1.48 | 1.38 |  | finite=1, r2=0.908983, rmse=4.80504 | - | r2=0.909, rmse=4.805 | ok |
| perceptron | taxi | algos | 1433 | 1487 | sklearn-cpu | 1081 | 1.33 | 1.38 |  | accuracy=0.76219 | - | accuracy=0.7505 | ok |
| louvain | istella | algos | 1880 | 1883 | networkx-cpu | 1376 | 1.37 | 1.37 |  | modularity=0.911187, n_communities=40 | - | modularity=0.9085, n_communities=40 | ok |
| lu-solve | synthetic | algos | 1433 | 685 | torch-gpu | 505 | 2.83 | 1.36 |  | relative_residual=3.2563e-06 | - | relative_residual=8.234e-07 | ok |
| knn | istella | classical | 766 | 764 | sklearn-cpu | 565 | 1.35 | 1.35 |  | recall_at_k=0.97625, rows_with_repeated_ids=0 | - | recall_at_k=1, rows_with_repeated_ids=0 | ok |
| resample | taxi | algos | 68.5 | 69.4 | sklearn-cpu | 51.9 | 1.32 | 1.34 |  | max_mean_shift_over_std=0.0029166 | - | max_mean_shift_over_std=0.002257 | ok |
| ivf-pq | istella | algos | 10146 | 9291 | faiss-cpu | 7008 | 1.45 | 1.33 |  | recall_at_10=0.550825 | - | recall_at_10=0.803 | ok |
| pa-reg | taxi | algos | 1511 | 1565 | sklearn-cpu | 1182 | 1.28 | 1.32 |  | finite=1, r2=0.900533, rmse=5.02315 | - | r2=0.7951, rmse=7.209 | ok |
| samba-forward | bytes | neural | 55.8 | 55.8 | torch-eager-fp32 | 42.5 | 1.31 | 1.31 |  | mean_nll=5.636 | mean_nll=5.636 | mean_nll=5.636 | ok |
| kernel-shap | taxi | algos | 1295 | 1206 | shap-cpu | 920 | 1.41 | 1.31 |  | rel_error_vs_exact=1.73139e-08 | - | rel_error_vs_exact=6.394e-14 | ok |
| ivf-filter | istella | algos | 10114 | 9304 | faiss-cpu | 7234 | 1.40 | 1.29 |  | recall_at_10=0.60925 | - | recall_at_10=0.8419 | ok |
| pa-clf | taxi | algos | 1503 | 1547 | sklearn-cpu | 1215 | 1.24 | 1.27 |  | accuracy=0.76036 | - | accuracy=0.7447 | ok |
| ivf-refine | istella | algos | 10124 | 9295 | faiss-cpu | 7322 | 1.38 | 1.27 |  | recall_at_10=0.809175 | - | recall_at_10=0.9934 | ok |
| gaussian-rp | taxi | algos | 3.8 | 2.0 | sklearn-cpu | 1.6 | 2.36 | 1.26 |  | mean_abs_distortion=0.345752 | - | mean_abs_distortion=0.3398 | ok |
| knn-imputer | istella | algos | 31.7 | 32.5 | sklearn-cpu (fill) | 26.0 | 1.22 | 1.25 |  | masked_rmse=323953 | - | masked_rmse=9.78e+05 | ok |
| multinomial-nb | taxi | algos | 16.7 | 48.7 | sklearn-cpu | 40.8 | 0.41 | 1.20 | FLIP slower | accuracy=0.72316, logloss=0.590725 | - | accuracy=0.7232, logloss=0.5907 | ok |
| auto-theta | taxi-hourly | algos | 4770 | 4769 | statsforecast-cpu | 4032 | 1.18 | 1.18 |  | forecast_rmse=49.0546 | - | forecast_rmse=49.27 | ok |
| resample | istella | algos | 390 | 394 | sklearn-cpu | 336 | 1.16 | 1.17 |  | max_mean_shift_over_std=0.00320274 | - | max_mean_shift_over_std=0.002552 | ok |
| theta | synthetic | algos | 120 | 121 | statsmodels-cpu | 104 | 1.16 | 1.16 |  | forecast_rmse=1.43661 | - | forecast_rmse=1.435 | ok |
| complement-nb | taxi | algos | 17.4 | 47.2 | sklearn-cpu | 40.9 | 0.43 | 1.15 | FLIP slower | accuracy=0.67802, logloss=0.715492 | - | accuracy=0.678, logloss=0.7155 | ok |
| gmm | istella | classical2 | 9155 | 8836 | sklearn-cpu | 7852 | 1.17 | 1.13 |  | bic=-3.85139e+07, mean_log_likelihood=200.794, n_iter=24 | - | bic=-3.901e+07, mean_log_likelihood=200.8, n_iter=30 | ok |
| ridge-clf | taxi | algos | 86.9 | 86.9 | sklearn-cpu | 77.9 | 1.12 | 1.12 |  | accuracy=0.76357 | - | accuracy=0.7636 | ok |
| target-encoder | taxi | algos | 407 | 163 | sklearn-cpu | 147 | 2.76 | 1.11 |  | - | - | - | ok |
| umap | istella | classical2 | 3349 | 1756 | umap-learn-cpu-unseeded | 1587 | 2.11 | 1.11 |  | trustworthiness_k15=0.977793 | - | trustworthiness_k15=0.9777 | ok |
| knn-imputer | taxi | algos | 2.2 | 1.8 | sklearn-cpu | 1.7 | 1.32 | 1.10 |  | masked_rmse=6.1517 | - | masked_rmse=5.257 | ok |
| ard | taxi | algos | 17.4 | 19.1 | sklearn-cpu | 17.5 | 1.00 | 1.09 | FLIP slower | finite=1, r2=0.909193, rmse=4.79951 | - | r2=0.9092, rmse=4.8 | ok |
| lle | taxi | algos | 3951 | 1360 | sklearn-cpu | 1247 | 3.17 | 1.09 |  | trustworthiness_k15=0.826083 | - | trustworthiness_k15=0.7708 | ok |
| nystroem | taxi | classical2 | 659 | 261 | sklearn-cpu | 246 | 2.68 | 1.06 |  | kernel_rel_error=0.0456109 | - | kernel_rel_error=0.04437 | ok |
| multilabel-binarizer | istella | algos | 173 | 150 | sklearn-cpu | 143 | 1.20 | 1.05 |  | - | - | - | ok |
| label-binarizer | istella | algos | 46.6 | 33.0 | sklearn-cpu | 31.7 | 1.47 | 1.04 |  | - | - | - | ok |
| lasso-lars | istella | algos | 287 | 238 | sklearn-cpu | 228 | 1.26 | 1.04 |  | finite=1, r2=0.31033, rmse=0.693715 | - | r2=0.3111, rmse=0.6933 | ok |
| svc | taxi | classical | 2269 | 2546 | sklearn-cpu | 2457 | 0.92 | 1.04 | FLIP slower | accuracy=0.7675, n_support=5527 | - | accuracy=0.7675, n_support=5672 | ok |
| select-mutual-info-reg | taxi | algos | 2796 | 2781 | sklearn-cpu | 2715 | 1.03 | 1.02 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| ivf | istella | classical2 | 5065 | 5005 | faiss-cpu | 5025 | 1.01 | 1.00 | FLIP faster | - | - | - | ok |
| randomized-svd | taxi | algos | 206 | 161 | sklearn-cpu | 163 | 1.26 | 0.99 | FLIP faster | relative_reconstruction_error=0.0271968 | - | relative_reconstruction_error=0.0272 | ok |
| dart-reg | istella | algos | 39168 | 31757 | lightgbm-cpu | 32389 | 1.21 | 0.98 | FLIP faster | finite=1, r2=0.550733, rmse=0.559903 | - | r2=0.5647, rmse=0.5512 | ok |
| nmf | istella | algos | 7483 | 6921 | sklearn-cpu | 7059 | 1.06 | 0.98 | FLIP faster | relative_reconstruction_error=0.325174 | - | relative_reconstruction_error=0.3254 | ok |
| maxabs-scaler | istella | algos | 122 | 95.1 | sklearn-cpu | 99.7 | 1.22 | 0.95 | FLIP faster | - | - | - | ok |
| bisecting-kmeans | istella | algos | 2351 | 1352 | sklearn-cpu | 1428 | 1.65 | 0.95 | FLIP faster | n_clusters=8, silhouette=0.118345 | - | ari_vs_ours=0.6929, n_clusters=8, silhouette=0.09641 | ok |
| dart | istella | algos | 39525 | 31833 | lightgbm-cpu | 35425 | 1.12 | 0.90 | FLIP faster | accuracy=0.94872, logloss=0.134126 | - | accuracy=0.9519, logloss=0.1237 | ok |
| mlp-reg | taxi | algos | 8701 | 8701 | sklearn-cpu | 9959 | 0.87 | 0.87 |  | r2=0.932, rmse=4.154 | r2=0.932, rmse=4.154 | r2=0.9296, rmse=4.226 | ok |
| ivf-sq | istella | algos | 4534 | 4427 | faiss-cpu | 5135 | 0.88 | 0.86 |  | recall_at_10=0.728025 | - | recall_at_10=0.5913 | ok |
| minmax-scaler | taxi | algos | 11.8 | 15.1 | sklearn-cpu | 17.6 | 0.67 | 0.86 |  | - | - | - | ok |
| lof | istella | algos | 10899 | 8614 | sklearn-cpu | 10177 | 1.07 | 0.85 | FLIP faster | fraction_flagged=0.03361 | - | fraction_flagged=0.03361, jaccard_vs_sklearn=1 | ok |
| mlp-reg | istella | algos | 11542 | 11542 | sklearn-cpu | 13651 | 0.85 | 0.85 |  | r2=0.5264, rmse=0.5749 | r2=0.5264, rmse=0.5749 | r2=0.5248, rmse=0.5758 | ok |
| target-encoder | istella | algos | 411 | 192 | sklearn-cpu | 229 | 1.80 | 0.84 | FLIP faster | - | - | - | ok |
| maxabs-scaler | taxi | algos | 8.0 | 10.3 | sklearn-cpu | 12.4 | 0.65 | 0.84 |  | - | - | - | ok |
| kernel-ridge | taxi | classical2 | 652 | 662 | sklearn-cpu | 794 | 0.82 | 0.83 |  | finite=1, r2=0.726543, rmse=8.33037 | - | r2=0.7265, rmse=8.33 | ok |
| label-spreading | istella | algos | 10856 | 8457 | sklearn-cpu | 10273 | 1.06 | 0.82 | FLIP faster | accuracy=0.90445 | - | accuracy=0.9044 | ok |
| louvain | taxi | algos | 636 | 633 | networkx-cpu | 781 | 0.81 | 0.81 |  | modularity=0.941953, n_communities=58 | - | modularity=0.9408, n_communities=56 | ok |
| svd | taxi | algos | 51.5 | 41.9 | torch-gpu | 51.8 | 1.00 | 0.81 |  | max_rel_singular_value_error=9.29813e-07, relative_reconstruction_error_100k_rows=1.82966e-06 | - | max_rel_singular_value_error=2.225e-06, relative_reconstruction_error_100k_rows=1.351e-05 | ok |
| gpc | taxi | classical2 | 589 | 589 | sklearn-cpu | 729 | 0.81 | 0.81 |  | accuracy=0.761, logloss=0.5413, nonfinite_proba_rows=0 | accuracy=0.761, logloss=0.5413, nonfinite_proba_rows=0 | accuracy=0.761, logloss=0.5414, nonfinite_proba_rows=0 | ok |
| mlp-clf | istella | algos | 11568 | 11568 | sklearn-cpu | 14327 | 0.81 | 0.81 |  | accuracy=0.9443, logloss=0.137 | accuracy=0.9443, logloss=0.137 | accuracy=0.9438, logloss=0.1368 | ok |
| cross-val-score | istella | algos | 6667 | 5665 | sklearn-cpu | 7051 | 0.95 | 0.80 |  | mean_r2=0.334732 | - | mean_r2=0.3198 | ok |
| kernel-ridge | istella | classical2 | 637 | 646 | sklearn-cpu | 836 | 0.76 | 0.77 |  | finite=1, r2=0.385427, rmse=0.646407 | - | r2=0.3854, rmse=0.6464 | ok |
| iterative-imputer | taxi | algos | 1760 | 807 | sklearn-cpu | 1049 | 1.68 | 0.77 | FLIP faster | masked_rmse=4.69385 | - | masked_rmse=4.694 | ok |
| ivf-rabitq | istella | algos | 4292 | 4203 | faiss-cpu | 5474 | 0.78 | 0.77 |  | recall_at_10=0.125125 | - | recall_at_10=0.05045 | ok |
| rnn-reg | taxi-hourly | algos | 1571 | 1571 | torch-eager-fp32 | 2050 | 0.77 | 0.77 |  | r2=0.7388, rmse=0.5543 | r2=0.7388, rmse=0.5543 | r2=0.7388, rmse=0.5543 | ok |
| mlp-clf | taxi | algos | 8718 | 8718 | sklearn-cpu | 11431 | 0.76 | 0.76 |  | accuracy=0.7678, logloss=0.5305 | accuracy=0.7678, logloss=0.5305 | accuracy=0.7678, logloss=0.5304 | ok |
| connected-components | istella | algos | 5.4 | 5.3 | networkx-cpu | 7.0 | 0.77 | 0.76 |  | n_components=81 | - | n_components=81 | ok |
| perceptron | istella | algos | 4294 | 4340 | sklearn-cpu | 5738 | 0.75 | 0.76 |  | accuracy=0.89569 | - | accuracy=0.8961 | ok |
| rnn-clf | synthetic | algos | 1568 | 1568 | torch-eager-fp32 | 2083 | 0.75 | 0.75 |  | accuracy=0.9536, logloss=0.1037 | accuracy=0.9536, logloss=0.1037 | accuracy=0.9536 | ok |
| rnn-reg | synthetic | algos | 1540 | 1540 | torch-eager-fp32 | 2057 | 0.75 | 0.75 |  | r2=0.9774, rmse=0.1744 | r2=0.9774, rmse=0.1744 | r2=0.9774, rmse=0.1744 | ok |
| enet-cv | taxi | algos | 163 | 158 | sklearn-cpu | 211 | 0.77 | 0.75 |  | finite=1, r2=0.909002, rmse=4.80454 | - | r2=0.909, rmse=4.804 | ok |
| pca | taxi | classical | 86.6 | 87.9 | sklearn-cpu | 120 | 0.72 | 0.73 |  | explained_variance_ratio_sum=0.999996 | - | explained_variance_ratio_sum=1 | ok |
| rnn-clf | taxi-hourly | algos | 1529 | 1529 | torch-eager-fp32 | 2091 | 0.73 | 0.73 |  | accuracy=0.8681, logloss=0.3049 | accuracy=0.8681, logloss=0.3049 | accuracy=0.8681 | ok |
| incremental-pca | taxi | algos | 160 | 100 | sklearn-cpu | 138 | 1.16 | 0.73 | FLIP faster | explained_variance_fraction=0.999995 | - | explained_variance_fraction=1 | ok |
| lasso-cv | taxi | algos | 156 | 158 | sklearn-cpu | 226 | 0.69 | 0.70 |  | finite=1, r2=0.909059, rmse=4.80305 | - | r2=0.909, rmse=4.804 | ok |
| sparse-rp | istella | algos | 54.5 | 17.3 | sklearn-cpu | 25.3 | 2.16 | 0.69 | FLIP faster | mean_abs_distortion=1.88338 | - | mean_abs_distortion=0.4743 | ok |
| mb-dict-learning | taxi | algos | 3639 | 3606 | sklearn-cpu | 5293 | 0.69 | 0.68 |  | component_sparsity=0, relative_reconstruction_error=0.496686 | - | component_sparsity=0, relative_reconstruction_error=0.477 | ok |
| lstsq | taxi | algos | 28.3 | 35.2 | numpy-cpu | 52.2 | 0.54 | 0.67 |  | relative_residual=0.756366 | - | relative_residual=0.7564 | ok |
| connected-components | taxi | algos | 9.5 | 4.9 | networkx-cpu | 7.3 | 1.29 | 0.67 | FLIP faster | n_components=588 | - | n_components=588 | ok |
| dynamic-theta | taxi-hourly | algos | 511 | 514 | statsforecast-cpu | 768 | 0.67 | 0.67 |  | forecast_rmse=49.1012 | - | forecast_rmse=49.27 | ok |
| select-d | taxi-hourly | algos | 4.8 | 2.2 | statsmodels-cpu | 3.4 | 1.42 | 0.66 | FLIP faster | - | - | d_agreement_vs_statsmodels=1 | ok |
| dynamic-optimized-theta | synthetic | algos | 625 | 625 | statsforecast-cpu | 959 | 0.65 | 0.65 |  | forecast_rmse=1.43628 | - | forecast_rmse=1.436 | ok |
| kmeans | istella | classical | 2072 | 2180 | sklearn-cpu | 3354 | 0.62 | 0.65 |  | inertia=6.05072e+17, inertia_over_ours=1, n_iter=33 | - | inertia=5.959e+17, inertia_over_ours=0.9848, n_iter=36 | ok |
| gpc | istella | classical2 | 798 | 798 | sklearn-cpu | 1230 | 0.65 | 0.65 |  | accuracy=0.9013, logloss=0.2326, nonfinite_proba_rows=0 | accuracy=0.9013, logloss=0.2326, nonfinite_proba_rows=0 | accuracy=0.9013, logloss=0.2326, nonfinite_proba_rows=0 | ok |
| gaussian-rp | istella | algos | 53.8 | 15.6 | sklearn-cpu | 24.3 | 2.21 | 0.64 | FLIP faster | mean_abs_distortion=0.680693 | - | mean_abs_distortion=0.178 | ok |
| auto-theta | synthetic | algos | 1225 | 1227 | statsforecast-cpu | 1953 | 0.63 | 0.63 |  | forecast_rmse=1.43891 | - | forecast_rmse=1.438 | ok |
| als | taxi-zones | algos | 18651 | 18423 | implicit-cpu | 30318 | 0.62 | 0.61 |  | recall_at_10=0.0539953 | - | recall_at_10=0.05679 | ok |
| label-propagation | istella | algos | 15655 | 8828 | sklearn-cpu | 14552 | 1.08 | 0.61 | FLIP faster | accuracy=0.9055 | - | accuracy=0.9055 | ok |
| svd | istella | algos | 1974 | 1478 | torch-gpu | 2473 | 0.80 | 0.60 |  | max_rel_singular_value_error=41531.1, relative_reconstruction_error_100k_rows=3.84493e-05 | - | max_rel_singular_value_error=4.945e+06, relative_reconstruction_error_100k_rows=0.0005905 | ok |
| knn-clf | istella | classical2 | 171 | 174 | sklearn-cpu | 293 | 0.58 | 0.59 |  | accuracy=0.92625 | - | accuracy=0.9263 | ok |
| spectral | taxi | classical2 | 1124 | 544 | sklearn-cpu | 923 | 1.22 | 0.59 | FLIP faster | n_clusters=8, silhouette=0.0399104 | - | ari_vs_ours=0.5823, n_clusters=8, silhouette=0.08989 | ok |
| sgd-ocsvm | istella | algos | 107029 | 3327 | sklearn-cpu | 5734 | 18.67 | 0.58 | FLIP faster | fraction_flagged=0.00044 | - | fraction_flagged=0.09334, jaccard_vs_sklearn=1 | ok |
| knn-reg | istella | classical2 | 164 | 165 | sklearn-cpu | 285 | 0.58 | 0.58 |  | finite=1, r2=0.418145, rmse=0.625388 | - | r2=0.4181, rmse=0.6254 | ok |
| bisecting-kmeans | taxi | algos | 296 | 221 | sklearn-cpu | 383 | 0.77 | 0.58 |  | n_clusters=8, silhouette=0.155357 | - | ari_vs_ours=0.5023, n_clusters=8, silhouette=0.128 | ok |
| iterative-imputer | istella | algos | 7808 | 5895 | sklearn-cpu | 10351 | 0.75 | 0.57 |  | masked_rmse=799013 | - | masked_rmse=8.024e+05 | ok |
| gru-reg | taxi-hourly | algos | 1727 | 1727 | torch-eager-fp32 | 3061 | 0.56 | 0.56 |  | r2=0.7482, rmse=0.5442 | r2=0.7482, rmse=0.5442 | r2=0.7482, rmse=0.5442 | ok |
| gru-reg | synthetic | algos | 1720 | 1720 | torch-eager-fp32 | 3051 | 0.56 | 0.56 |  | r2=0.9819, rmse=0.1557 | r2=0.9819, rmse=0.1557 | r2=0.9819, rmse=0.1557 | ok |
| select-f-classif | taxi | algos | 216 | 22.9 | sklearn-cpu | 41.1 | 5.26 | 0.56 | FLIP faster | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| lof | taxi | algos | 2209 | 1875 | sklearn-cpu | 3405 | 0.65 | 0.55 |  | fraction_flagged=0.00896 | - | fraction_flagged=0.00896, jaccard_vs_sklearn=1 | ok |
| gru-clf | taxi-hourly | algos | 1722 | 1722 | torch-eager-fp32 | 3132 | 0.55 | 0.55 |  | accuracy=0.8657, logloss=0.3058 | accuracy=0.8657, logloss=0.3058 | accuracy=0.8657 | ok |
| optimized-theta | taxi-hourly | algos | 708 | 707 | statsforecast-cpu | 1290 | 0.55 | 0.55 |  | forecast_rmse=49.1509 | - | forecast_rmse=49.36 | ok |
| gru-clf | synthetic | algos | 1725 | 1725 | torch-eager-fp32 | 3162 | 0.55 | 0.55 |  | accuracy=0.9718, logloss=0.06584 | accuracy=0.9718, logloss=0.06584 | accuracy=0.9718 | ok |
| gpr | taxi | classical2 | 141 | 89.2 | sklearn-cpu | 164 | 0.86 | 0.54 |  | finite=1, mean_log_predictive_density=-311.458, r2=0.88963, rmse=5.04164 | - | mean_log_predictive_density=-311.5, r2=0.8896, rmse=5.042 | ok |
| umap | taxi | classical2 | 1320 | 777 | umap-learn-cpu-unseeded | 1448 | 0.91 | 0.54 |  | trustworthiness_k15=0.991778 | - | trustworthiness_k15=0.9895 | ok |
| select-d | synthetic | algos | 6.4 | 2.7 | statsmodels-cpu | 5.1 | 1.27 | 0.53 | FLIP faster | - | - | d_agreement_vs_statsmodels=1 | ok |
| optimized-theta | synthetic | algos | 397 | 397 | statsforecast-cpu | 749 | 0.53 | 0.53 |  | forecast_rmse=1.43886 | - | forecast_rmse=1.438 | ok |
| random-trees-embedding | istella | algos | 270 | 325 | sklearn-cpu | 625 | 0.43 | 0.52 |  | nonzeros_per_row=10, output_columns=209 | - | nonzeros_per_row=10, output_columns=251 | ok |
| select-chi2 | istella | algos | 318 | 146 | sklearn-cpu | 283 | 1.12 | 0.52 | FLIP faster | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| lda | text | algos | 36638 | 37231 | sklearn-cpu | 72657 | 0.50 | 0.51 |  | perplexity=266.719 | - | perplexity=266.9 | ok |
| select-f-classif | istella | algos | 336 | 150 | sklearn-cpu | 294 | 1.14 | 0.51 | FLIP faster | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| incremental-pca | istella | algos | 2905 | 2905 | sklearn-cpu | 5745 | 0.51 | 0.51 |  | explained_variance_fraction=1 | explained_variance_fraction=1 | explained_variance_fraction=1 | error |
| lle | istella | algos | 4044 | 1438 | sklearn-cpu | 2852 | 1.42 | 0.50 | FLIP faster | trustworthiness_k15=0.895277 | - | trustworthiness_k15=0.8491 | ok |
| select-f-regression | istella | algos | 269 | 134 | sklearn-cpu | 268 | 1.00 | 0.50 | FLIP faster | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| select-r-regression | istella | algos | 264 | 133 | sklearn-cpu | 268 | 0.98 | 0.50 |  | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| meanshift | istella | algos | 237 | 218 | sklearn-cpu | 444 | 0.53 | 0.49 |  | n_clusters=12, silhouette=0.403452 | - | ari_vs_ours=1, n_clusters=12, silhouette=0.4034 | ok |
| pa-clf | istella | algos | 4399 | 4429 | sklearn-cpu | 9215 | 0.48 | 0.48 |  | accuracy=0.92259 | - | accuracy=0.8905 | ok |
| kmeans | taxi | classical | 951 | 973 | sklearn-cpu | 2058 | 0.46 | 0.47 |  | inertia=3.09303e+08, inertia_over_ours=1, n_iter=91 | - | inertia=3.166e+08, inertia_over_ours=1.024, n_iter=100 | ok |
| isotonic | taxi | algos | 44.8 | 42.4 | sklearn-cpu | 90.5 | 0.50 | 0.47 |  | finite=1, r2=0.897069, rmse=5.10987 | - | r2=0.8971, rmse=5.11 | ok |
| dict-learning | taxi | algos | 5472 | 5361 | sklearn-cpu | 11561 | 0.47 | 0.46 |  | component_sparsity=0, relative_reconstruction_error=0.459169 | - | component_sparsity=0, relative_reconstruction_error=0.4606 | ok |
| select-r-regression | taxi | algos | 146 | 14.6 | sklearn-cpu | 31.8 | 4.59 | 0.46 | FLIP faster | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| ridge-clf | istella | algos | 3207 | 3171 | sklearn-cpu | 7044 | 0.46 | 0.45 |  | accuracy=0.91054 | - | accuracy=0.9105 | ok |
| dict-learning | istella | algos | 7442 | 7353 | sklearn-cpu | 16442 | 0.45 | 0.45 |  | component_sparsity=0.0863636, relative_reconstruction_error=0.652121 | - | component_sparsity=0.08636, relative_reconstruction_error=0.6521 | ok |
| ols | taxi | classical | 303 | 121 | sklearn-cpu | 274 | 1.10 | 0.44 | FLIP faster | finite=1, r2=0.908837, rmse=4.69648 | - | r2=0.7248, rmse=8.159 | ok |
| garch | taxi-hourly | algos | 772 | 53.1 | arch-cpu | 125 | 6.19 | 0.43 | FLIP faster | mean_llf=-1132.71 | - | mean_llf=-1130 | ok |
| standard-scaler | taxi | algos | 12.8 | 19.3 | sklearn-cpu | 45.4 | 0.28 | 0.42 |  | - | - | - | ok |
| select-f-regression | taxi | algos | 146 | 13.2 | sklearn-cpu | 31.3 | 4.66 | 0.42 | FLIP faster | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| dynamic-theta | synthetic | algos | 139 | 139 | statsforecast-cpu | 333 | 0.42 | 0.42 |  | forecast_rmse=1.43726 | - | forecast_rmse=1.437 | ok |
| bpe-encode | enwik8 | algos | 53.8 | 55.9 | hf-tokenizers-cpu | 134 | 0.40 | 0.42 |  | documents_equal_to_ours=1, tokens=1.45732e+06 | - | documents_equal_to_ours=1, tokens=1.457e+06 | ok |
| gaussian-nb | istella | algos | 141 | 148 | sklearn-cpu | 361 | 0.39 | 0.41 |  | accuracy=0.87657, logloss=3.57442 | - | accuracy=0.8765, logloss=3.417 | ok |
| als | text | algos | - | 127744 | implicit-cpu (fill) | 321338 | - | 0.40 |  | recall_at_10=0.548738 | - | recall_at_10=0.5482 | ok |
| ols | istella | classical | 3070 | 1281 | sklearn-cpu | 3245 | 0.95 | 0.39 |  | finite=1, r2=0.332506, rmse=0.68174 | - | r2=0.001881, rmse=0.8337 | ok |
| ocsvm | taxi | algos | 383 | 70.8 | sklearn-cpu | 181 | 2.12 | 0.39 | FLIP faster | fraction_flagged=0.1361 | - | fraction_flagged=0.1361, jaccard_vs_sklearn=1 | ok |
| poly-count-sketch | istella | algos | 0.4 | 1.2 | sklearn-cpu | 3.1 | 0.12 | 0.39 |  | kernel_rel_error=0.040849 | - | kernel_rel_error=0.04085 | ok |
| lasso-cv | istella | algos | 2125 | 2077 | sklearn-cpu | 5483 | 0.39 | 0.38 |  | finite=1, r2=0.310329, rmse=0.693715 | - | r2=0.3108, rmse=0.6935 | ok |
| pls | taxi | algos | 108 | 90.6 | sklearn-cpu | 242 | 0.45 | 0.37 |  | finite=1, r2=0.905216, rmse=4.90347 | - | r2=0.9052, rmse=4.904 | ok |
| pls-canonical | taxi | algos | 123 | 100 | sklearn-cpu | 269 | 0.46 | 0.37 |  | mean_canonical_corr=0.559206 | - | mean_canonical_corr=0.5592 | ok |
| tsne | istella | algos | 3214 | 7076 | sklearn-cpu | 19429 | 0.17 | 0.36 |  | trustworthiness_k15=0.992206 | - | trustworthiness_k15=0.992 | ok |
| pa-reg | istella | algos | 4406 | 4443 | sklearn-cpu | 12345 | 0.36 | 0.36 |  | finite=1, r2=0.294299, rmse=0.701731 | - | r2=-0.1282, rmse=0.8872 | ok |
| mamba3-forward | gaussian | neural | 29.0 | 29.0 | torch-eager-fp32 | 80.7 | 0.36 | 0.36 |  | - | - | - | ok |
| tsne | taxi | algos | 2340 | 6162 | sklearn-cpu | 17507 | 0.13 | 0.35 |  | trustworthiness_k15=0.99889 | - | trustworthiness_k15=0.9989 | ok |
| ivf-refine | taxi | algos | 1535 | 1535 | faiss-cpu | 4363 | 0.35 | 0.35 |  | recall_at_10=0.9997 | recall_at_10=0.9997 | recall_at_10=0.9992 | ok |
| nystroem | istella | classical2 | 610 | 178 | sklearn-cpu | 509 | 1.20 | 0.35 | FLIP faster | kernel_rel_error=0.0334304 | - | kernel_rel_error=0.03896 | ok |
| rfe | taxi | algos | 100 | 109 | sklearn-cpu | 313 | 0.32 | 0.35 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| ivf-pq | taxi | algos | 1533 | 1495 | faiss-cpu | 4367 | 0.35 | 0.34 |  | recall_at_10=0.973225 | - | recall_at_10=0.9801 | ok |
| ivf-filter | taxi | algos | 1519 | 1492 | faiss-cpu | 4368 | 0.35 | 0.34 |  | recall_at_10=0.980075 | - | recall_at_10=0.982 | ok |
| skewed-chi2 | istella | algos | 7.5 | 1.3 | sklearn-cpu | 3.7 | 1.99 | 0.34 | FLIP faster | kernel_rel_error=0.671898 | - | kernel_rel_error=0.6719 | ok |
| mb-sparse-pca | istella | algos | 877 | 677 | sklearn-cpu | 2006 | 0.44 | 0.34 |  | component_sparsity=0.130682, relative_reconstruction_error=0.705389 | - | component_sparsity=0.1307, relative_reconstruction_error=0.7054 | ok |
| pls | istella | algos | 622 | 654 | sklearn-cpu | 2018 | 0.31 | 0.32 |  | finite=1, r2=0.28987, rmse=0.70393 | - | r2=0.2899, rmse=0.7039 | ok |
| sparse-pca | istella | algos | 3310 | 3201 | sklearn-cpu | 9904 | 0.33 | 0.32 |  | component_sparsity=0.305682, relative_reconstruction_error=0.75021 | - | component_sparsity=0.3057, relative_reconstruction_error=0.7502 | ok |
| categorical-nb | istella | algos | 18.1 | 27.0 | sklearn-cpu | 84.4 | 0.21 | 0.32 |  | accuracy=0.83885, logloss=0.412625 | - | accuracy=0.8388, logloss=0.4126 | ok |
| tsvd | istella | classical2 | 424 | 447 | sklearn-cpu | 1408 | 0.30 | 0.32 |  | explained_variance_ratio_sum=0.999992, relative_reconstruction_error=0.00255392 | - | explained_variance_ratio_sum=1, relative_reconstruction_error=0.000122 | ok |
| select-chi2 | taxi | algos | 209 | 18.5 | sklearn-cpu | 58.5 | 3.58 | 0.32 | FLIP faster | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| qr | taxi | algos | 42.3 | 35.3 | numpy-cpu | 114 | 0.37 | 0.31 |  | relative_gram_difference=5.57879e-07 | - | relative_gram_difference=3.024e-08 | ok |
| isomap | istella | algos | 4088 | 4066 | sklearn-cpu | 13252 | 0.31 | 0.31 |  | trustworthiness_k15=0.853294 | - | trustworthiness_k15=0.8533 | ok |
| cca | taxi | algos | 184 | 141 | sklearn-cpu | 464 | 0.40 | 0.30 |  | mean_canonical_corr=0.576863 | - | mean_canonical_corr=0.5769 | ok |
| standard-scaler | istella | algos | 134 | 144 | sklearn-cpu | 478 | 0.28 | 0.30 |  | - | - | - | ok |
| enet-cv | istella | algos | 2122 | 2082 | sklearn-cpu | 6979 | 0.30 | 0.30 |  | finite=1, r2=0.316583, rmse=0.690563 | - | r2=0.3173, rmse=0.6902 | ok |
| nearest-centroid | istella | algos | 289 | 150 | sklearn-cpu | 529 | 0.55 | 0.28 |  | accuracy=0.85261, logloss=4.29922 | - | accuracy=0.8526, logloss=4.118 | ok |
| spectral | istella | classical2 | 961 | 444 | sklearn-cpu | 1595 | 0.60 | 0.28 |  | n_clusters=8, silhouette=0.147668 | - | ari_vs_ours=0.9998, n_clusters=8, silhouette=0.1477 | ok |
| pls-canonical | istella | algos | 1560 | 1117 | sklearn-cpu | 4065 | 0.38 | 0.27 |  | mean_canonical_corr=0.87534 | - | mean_canonical_corr=0.8753 | ok |
| garch | synthetic | algos | 626 | 28.2 | arch-cpu | 106 | 5.93 | 0.27 | FLIP faster | mean_llf=-1938.22 | - | mean_llf=-1938 | ok |
| label-spreading | taxi | algos | 2058 | 1762 | sklearn-cpu | 6648 | 0.31 | 0.27 |  | accuracy=0.6764 | - | accuracy=0.6764 | ok |
| categorical-nb | taxi | algos | 18.8 | 20.7 | sklearn-cpu | 78.3 | 0.24 | 0.26 |  | accuracy=0.76585, logloss=0.538866 | - | accuracy=0.7659, logloss=0.5389 | ok |
| bayesian-gmm | istella | algos | 32468 | 30214 | sklearn-cpu (fill) | 116803 | 0.28 | 0.26 |  | mean_log_likelihood=175.322 | - | - | ok |
| factor-analysis | istella | algos | 9521 | 9052 | sklearn-cpu | 35135 | 0.27 | 0.26 |  | mean_log_likelihood=99.4872 | - | mean_log_likelihood=98.12 | ok |
| fastica | taxi | algos | 145 | 107 | sklearn-cpu | 419 | 0.35 | 0.26 |  | mean_abs_excess_kurtosis=13.2051 | - | mean_abs_excess_kurtosis=13.77 | ok |
| nearest-centroid | taxi | algos | 192 | 24.7 | sklearn-cpu | 96.9 | 1.98 | 0.26 | FLIP faster | accuracy=0.66675, logloss=0.782162 | - | accuracy=0.6667, logloss=0.7817 | ok |
| gaussian-nb | taxi | algos | 18.9 | 19.1 | sklearn-cpu | 76.2 | 0.25 | 0.25 |  | accuracy=0.71982, logloss=1.13225 | - | accuracy=0.7199, logloss=1.134 | ok |
| lstsq | istella | algos | 915 | 764 | numpy-cpu | 3086 | 0.30 | 0.25 |  | relative_residual=0.849956 | - | relative_residual=0.8733 | ok |
| ridge | istella | classical2 | 1447 | 1443 | sklearn-cpu | 5981 | 0.24 | 0.24 |  | finite=1, r2=0.328682, rmse=0.684423 | - | r2=0.3287, rmse=0.6844 | ok |
| ridge-cv | istella | algos | 31606 | 31537 | sklearn-cpu | 134463 | 0.24 | 0.23 |  | finite=1, r2=0.328684, rmse=0.684422 | - | r2=0.3287, rmse=0.6844 | ok |
| bernoulli-nb | taxi | algos | 19.1 | 21.6 | sklearn-cpu | 92.5 | 0.21 | 0.23 |  | accuracy=0.75556, logloss=0.557803 | - | accuracy=0.7556, logloss=0.5578 | ok |
| isomap | taxi | algos | 3202 | 3165 | sklearn-cpu | 13665 | 0.23 | 0.23 |  | trustworthiness_k15=0.771828 | - | trustworthiness_k15=0.7718 | ok |
| nmf | taxi | algos | 863 | 781 | sklearn-cpu | 3419 | 0.25 | 0.23 |  | relative_reconstruction_error=0.0911559 | - | relative_reconstruction_error=0.09115 | ok |
| mamba1-forward | gaussian | neural | 21.2 | 21.2 | torch-eager-fp32 | 93.3 | 0.23 | 0.23 |  | - | - | - | ok |
| cross-val-score | taxi | algos | 317 | 283 | sklearn-cpu | 1282 | 0.25 | 0.22 |  | mean_r2=0.937956 | - | mean_r2=0.938 | ok |
| ocsvm | istella | algos | 463 | 152 | sklearn-cpu | 691 | 0.67 | 0.22 |  | fraction_flagged=0.0783 | - | fraction_flagged=0.0783, jaccard_vs_sklearn=1 | ok |
| mamba2-forward | gaussian | neural | 33.9 | 33.9 | torch-compile-bf16 | 154 | 0.22 | 0.22 |  | - | - | - | ok |
| label-encoder | istella | algos | 16.9 | 4.9 | sklearn-cpu | 22.5 | 0.75 | 0.22 |  | - | - | - | ok |
| multioutput-reg | istella | algos | 2976 | 2921 | sklearn-cpu | 13401 | 0.22 | 0.22 |  | r2=0.455327 | - | r2=0.4553 | ok |
| label-propagation | taxi | algos | 6835 | 2239 | sklearn-cpu | 10366 | 0.66 | 0.22 |  | accuracy=0.7016 | - | accuracy=0.7016 | ok |
| ets | synthetic | classical2 | 188 | 184 | statsmodels-cpu | 860 | 0.22 | 0.21 |  | forecast_rmse=0.984392, insample_rmse=0.990971 | - | forecast_rmse=0.9844, insample_rmse=0.9918 | ok |
| croston-optimized | taxi-hourly | algos | 12.2 | 12.6 | statsforecast-cpu | 60.5 | 0.20 | 0.21 |  | forecast_rmse=1.39825 | - | forecast_rmse=1.398 | ok |
| label-encoder | taxi | algos | 11.5 | 5.8 | sklearn-cpu | 29.0 | 0.40 | 0.20 |  | - | - | - | ok |
| gpr | istella | classical2 | 167 | 104 | sklearn-cpu | 523 | 0.32 | 0.20 |  | finite=1, mean_log_predictive_density=-9.28575, r2=0.235346, rmse=0.760439 | - | mean_log_predictive_density=-9.287, r2=0.2354, rmse=0.7604 | ok |
| agglomerative | taxi | classical2 | 71.2 | 44.2 | sklearn-cpu | 224 | 0.32 | 0.20 |  | n_clusters=8, silhouette=0.685524 | - | ari_vs_ours=1, n_clusters=8, silhouette=0.6855 | ok |
| kbins | taxi | algos | 183 | 21.0 | sklearn-cpu | 108 | 1.68 | 0.19 | FLIP faster | - | - | - | ok |
| affinity-prop | istella | algos | 1247 | 1113 | sklearn-cpu | 5734 | 0.22 | 0.19 |  | n_clusters=342, silhouette=0.0897635 | - | ari_vs_ours=1, n_clusters=342, silhouette=0.08976 | ok |
| robust-scaler | taxi | algos | 65.7 | 21.0 | sklearn-cpu | 109 | 0.60 | 0.19 |  | - | - | - | ok |
| lda-clf | istella | algos | 22056 | 711 | sklearn-cpu | 3739 | 5.90 | 0.19 | FLIP faster | accuracy=0.91166, logloss=0.247483 | - | accuracy=0.9011, logloss=0.4492 | ok |
| dart | taxi | algos | 7367 | 5011 | lightgbm-cpu | 26498 | 0.28 | 0.19 |  | accuracy=0.76834, logloss=0.52906 | - | accuracy=0.7682, logloss=0.5291 | ok |
| variance-threshold | taxi | algos | 125 | 9.5 | sklearn-cpu | 50.6 | 2.48 | 0.19 | FLIP faster | - | - | - | ok |
| croston-optimized | synthetic | algos | 12.7 | 11.5 | statsforecast-cpu | 62.3 | 0.20 | 0.18 |  | forecast_rmse=1.67554 | - | forecast_rmse=1.675 | ok |
| variance-threshold | istella | algos | 239 | 124 | sklearn-cpu | 688 | 0.35 | 0.18 |  | - | - | - | ok |
| poly-features | taxi | algos | 0.2 | 0.3 | sklearn-cpu | 1.6 | 0.15 | 0.18 |  | - | - | - | ok |
| dart-reg | taxi | algos | 6880 | 4970 | lightgbm-cpu | 27650 | 0.25 | 0.18 |  | finite=1, r2=0.925497, rmse=4.34734 | - | r2=0.9263, rmse=4.325 | ok |
| knn-clf | taxi | classical2 | 24.3 | 27.5 | sklearn-cpu | 154 | 0.16 | 0.18 |  | accuracy=0.74175 | - | accuracy=0.7418 | ok |
| svr | taxi | classical2 | 195 | 205 | sklearn-cpu | 1144 | 0.17 | 0.18 |  | finite=1, r2=0.767551, rmse=7.6804 | - | r2=0.7675, rmse=7.68 | ok |
| svc | istella | classical | 142 | 190 | sklearn-cpu | 1071 | 0.13 | 0.18 |  | accuracy=0.9222, n_support=2400 | - | accuracy=0.9222, n_support=2400 | ok |
| poisson | taxi | algos | - | 77.7 | sklearn-cpu (fill) | 442 | - | 0.18 |  | finite=1, r2=0.0357566, rmse=15.6398 | - | - | ok |
| tsvd | taxi | classical2 | 31.9 | 27.5 | sklearn-cpu | 157 | 0.20 | 0.17 |  | explained_variance_ratio_sum=0.999965, relative_reconstruction_error=0.00325707 | - | explained_variance_ratio_sum=1, relative_reconstruction_error=0.003257 | ok |
| complement-nb | text | algos | 208 | 44.6 | sklearn-cpu | 265 | 0.79 | 0.17 |  | accuracy=0.983067, logloss=0.559491 | - | accuracy=0.9831, logloss=0.5573 | ok |
| bernoulli-nb | istella | algos | 161 | 141 | sklearn-cpu | 869 | 0.19 | 0.16 |  | accuracy=0.79405, logloss=5.35063 | - | accuracy=0.7941, logloss=4.279 | ok |
| qda | taxi | algos | 210 | 22.8 | sklearn-cpu | 141 | 1.49 | 0.16 | FLIP faster | accuracy=0.72702, logloss=1.061 | - | accuracy=0.7272, logloss=1.059 | ok |
| tree-shap | istella | algos | 81.9 | 24.1 | lightgbm-cpu | 148 | 0.55 | 0.16 |  | max_additivity_error=1.17498e-06 | - | max_additivity_error=4.441e-15 | ok |
| sparse-pca | taxi | algos | 1940 | 1893 | sklearn-cpu | 11923 | 0.16 | 0.16 |  | component_sparsity=0.488636, relative_reconstruction_error=0.277226 | - | component_sparsity=0.4886, relative_reconstruction_error=0.2772 | ok |
| croston-sba | synthetic | algos | 4.8 | 5.4 | statsforecast-cpu | 34.5 | 0.14 | 0.16 |  | forecast_rmse=1.67446 | - | forecast_rmse=1.675 | ok |
| knn-reg | taxi | classical2 | 22.5 | 20.4 | sklearn-cpu | 131 | 0.17 | 0.16 |  | finite=1, r2=0.937323, rmse=3.84203 | - | r2=0.9373, rmse=3.842 | ok |
| qda | istella | algos | 19056 | 979 | sklearn-cpu | 6397 | 2.98 | 0.15 | FLIP faster | accuracy=0.86609, logloss=4.05287 | - | accuracy=0.8805, logloss=3.477 | ok |
| qr | istella | algos | 1823 | 1328 | numpy-cpu | 8840 | 0.21 | 0.15 |  | relative_gram_difference=1.53936e-07 | - | relative_gram_difference=2.473e-08 | ok |
| poly-features | istella | algos | 0.3 | 0.3 | sklearn-cpu | 2.1 | 0.15 | 0.15 |  | - | - | - | ok |
| ivf | taxi | classical2 | 560 | 555 | faiss-cpu | 3751 | 0.15 | 0.15 |  | - | - | - | ok |
| lda | taxi-zones | algos | 3065 | 3070 | sklearn-cpu | 20926 | 0.15 | 0.15 |  | perplexity=45.2218 | - | perplexity=44.9 | ok |
| gamma | taxi | algos | - | 52.7 | sklearn-cpu (fill) | 360 | - | 0.15 |  | finite=1, r2=-231.912, rmse=243.071 | - | - | ok |
| multinomial-nb | text | algos | 206 | 38.0 | sklearn-cpu | 266 | 0.78 | 0.14 |  | accuracy=0.983067, logloss=0.559529 | - | accuracy=0.9831, logloss=0.5573 | ok |
| arima | synthetic | classical2 | 132 | 51.9 | statsmodels-cpu | 364 | 0.36 | 0.14 |  | forecast_rmse=1.51552, insample_rmse=0.999342, mean_aic=5680.98, mean_llf=-2836.49 | - | forecast_rmse=1.515, insample_rmse=0.9993, mean_aic=5681, mean_llf=-2836 | ok |
| svr | istella | classical2 | 195 | 203 | sklearn-cpu | 1422 | 0.14 | 0.14 |  | finite=1, r2=0.318258, rmse=0.680816 | - | r2=0.3182, rmse=0.6808 | ok |
| lda-clf | taxi | algos | 292 | 26.0 | sklearn-cpu | 186 | 1.58 | 0.14 | FLIP faster | accuracy=0.76258, logloss=0.539749 | - | accuracy=0.7625, logloss=0.5398 | ok |
| lr-step | synthetic | algos | 11.8 | 11.8 | torch-cpu | 85.1 | 0.14 | 0.14 |  | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=0 | max_rel_diff_vs_ours=1.49e-08 | ok |
| ivf-sq | taxi | algos | 527 | 524 | faiss-cpu | 3783 | 0.14 | 0.14 |  | recall_at_10=0.934975 | - | recall_at_10=0.857 | ok |
| tweedie | taxi | algos | - | 45.1 | sklearn-cpu (fill) | 333 | - | 0.14 |  | finite=1, r2=-10.0744, rmse=53.0026 | - | - | ok |
| random-trees-embedding | taxi | algos | 48.6 | 58.9 | sklearn-cpu | 441 | 0.11 | 0.13 |  | nonzeros_per_row=10, output_columns=292 | - | nonzeros_per_row=10, output_columns=244 | ok |
| tree-shap | taxi | algos | 23.2 | 9.7 | lightgbm-cpu | 72.7 | 0.32 | 0.13 |  | max_additivity_error=3.85772e-05 | - | max_additivity_error=5.684e-13 | ok |
| huber | taxi | algos | 227 | 244 | sklearn-cpu | 1855 | 0.12 | 0.13 |  | finite=1, r2=0.900215, rmse=5.03117 | - | r2=0.9002, rmse=5.031 | ok |
| linearsvc | taxi | classical2 | 56.3 | 68.8 | sklearn-cpu | 528 | 0.11 | 0.13 |  | accuracy=0.76333 | - | accuracy=0.7636 | ok |
| ivf-rabitq | taxi | algos | 519 | 508 | faiss-cpu | 3984 | 0.13 | 0.13 |  | recall_at_10=0.110475 | - | recall_at_10=0.1263 | ok |
| prophet | synthetic | algos | 414 | 57.3 | prophet-cpu | 452 | 0.92 | 0.13 |  | forecast_rmse=1.01515 | - | forecast_rmse=1.015 | ok |
| hdbscan | taxi | classical | 3928 | 3846 | sklearn-cpu | 30753 | 0.13 | 0.13 |  | n_clusters=161, noise_fraction=0.14055, rows=100000 | - | n_clusters=160, noise_fraction=0.1422, rows=1e+05 | ok |
| cca | istella | algos | 10056 | 9028 | sklearn-cpu | 73061 | 0.14 | 0.12 |  | mean_canonical_corr=0.998053 | - | mean_canonical_corr=0.9996 | ok |
| bagging-clf | taxi | algos | 189 | 265 | sklearn-cpu | 2206 | 0.09 | 0.12 |  | accuracy=0.76749, logloss=0.531007 | - | accuracy=0.7677, logloss=0.5301 | ok |
| quantile-transformer | taxi | algos | 184 | 20.7 | sklearn-cpu | 174 | 1.06 | 0.12 | FLIP faster | - | - | - | ok |
| stacking-reg | taxi | algos | 794 | 713 | sklearn-cpu | 6007 | 0.13 | 0.12 |  | finite=1, r2=0.919718, rmse=4.51281 | - | r2=0.9325, rmse=4.137 | ok |
| bagging-reg | taxi | algos | 167 | 260 | sklearn-cpu | 2202 | 0.08 | 0.12 |  | finite=1, r2=0.916956, rmse=4.58978 | - | r2=0.9388, rmse=3.94 | ok |
| select-mutual-info | taxi | algos | 218 | 210 | sklearn-cpu | 1786 | 0.12 | 0.12 |  | n_selected=5 | - | jaccard_vs_sklearn=1, n_selected=5 | ok |
| multioutput-clf | taxi | algos | 67.2 | 80.1 | sklearn-cpu | 686 | 0.10 | 0.12 |  | accuracy=0.86356 | - | accuracy=0.8636 | ok |
| rfe | istella | algos | 1914 | 1846 | sklearn-cpu | 16177 | 0.12 | 0.11 |  | n_selected=110 | - | jaccard_vs_sklearn=1, n_selected=110 | ok |
| spectral-embedding | taxi | classical2 | 516 | 284 | sklearn-cpu | 2537 | 0.20 | 0.11 |  | trustworthiness_k15=0.884879 | - | trustworthiness_k15=0.898 | ok |
| stacking-clf | taxi | algos | 693 | 777 | sklearn-cpu | 7095 | 0.10 | 0.11 |  | accuracy=0.76792, logloss=0.536364 | - | accuracy=0.7553, logloss=0.5478 | ok |
| adaboost-reg | taxi | algos | 2090 | 1320 | sklearn-cpu | 12197 | 0.17 | 0.11 |  | finite=1, r2=0.679395, rmse=9.01824 | - | r2=0.5639, rmse=10.52 | ok |
| bayesian-ridge | istella | algos | 768 | 719 | sklearn-cpu | 6831 | 0.11 | 0.11 |  | finite=1, r2=0.317909, rmse=0.689893 | - | r2=-890.2, rmse=24.94 | ok |
| ard | istella | algos | 983 | 1039 | sklearn-cpu | 10421 | 0.09 | 0.10 |  | finite=1, r2=-0.124856, rmse=0.885949 | - | r2=0.3274, rmse=0.6851 | ok |
| croston-sba | taxi-hourly | algos | 3.2 | 4.7 | statsforecast-cpu | 47.9 | 0.07 | 0.10 |  | forecast_rmse=1.38668 | - | forecast_rmse=1.387 | ok |
| fastica | istella | algos | 1588 | 1588 | sklearn-cpu | 16796 | 0.09 | 0.09 |  | mean_abs_excess_kurtosis=356.3 | mean_abs_excess_kurtosis=356.3 | mean_abs_excess_kurtosis=922.5 | error |
| adaboost-clf | taxi | algos | 2399 | 2577 | sklearn-cpu | 27304 | 0.09 | 0.09 |  | accuracy=0.76523, logloss=0.543227 | - | accuracy=0.7654, logloss=0.5406 | ok |
| multioutput-clf | istella | algos | 1631 | 2107 | sklearn-cpu | 22340 | 0.07 | 0.09 |  | accuracy=0.959195 | - | accuracy=0.9592 | ok |
| croston | taxi-hourly | algos | 4.9 | 4.9 | statsforecast-cpu | 54.2 | 0.09 | 0.09 |  | forecast_rmse=1.39026 | - | forecast_rmse=1.39 | ok |
| logreg-cv | istella | algos | 6204 | 6372 | sklearn-cpu | 72840 | 0.09 | 0.09 |  | accuracy=0.9245, logloss=0.181393 | - | accuracy=0.9246, logloss=0.1814 | ok |
| logreg | istella | classical2 | 2533 | 2805 | sklearn-cpu | 32177 | 0.08 | 0.09 |  | accuracy=0.92459, logloss=0.181249, nonfinite_proba_rows=0 | - | accuracy=0.9245, logloss=0.1813, nonfinite_proba_rows=0 | ok |
| prophet | taxi-hourly | algos | 381 | 46.5 | prophet-cpu | 549 | 0.69 | 0.08 |  | forecast_rmse=32.0493 | - | forecast_rmse=32.03 | ok |
| ovr | taxi | algos | 113 | 123 | sklearn-cpu | 1461 | 0.08 | 0.08 |  | accuracy=0.47893 | - | accuracy=0.4789 | ok |
| affinity-prop | taxi | algos | 423 | 389 | sklearn-cpu | 4839 | 0.09 | 0.08 |  | n_clusters=272, silhouette=0.184644 | - | ari_vs_ours=1, n_clusters=272, silhouette=0.1846 | ok |
| bayesian-gmm | taxi | algos | 2506 | 316 | sklearn-cpu | 3978 | 0.63 | 0.08 |  | mean_log_likelihood=4.8956 | - | mean_log_likelihood=6.178 | ok |
| logreg-cv | taxi | algos | 231 | 139 | sklearn-cpu | 1758 | 0.13 | 0.08 |  | accuracy=0.76332, logloss=0.538985 | - | accuracy=0.7633, logloss=0.539 | ok |
| mds | istella | algos | 197 | 203 | sklearn-cpu | 2666 | 0.07 | 0.08 |  | trustworthiness_k15=0.586415 | - | trustworthiness_k15=0.5802 | ok |
| voting-reg | taxi | algos | 132 | 96.5 | sklearn-cpu | 1272 | 0.10 | 0.08 |  | finite=1, r2=0.919181, rmse=4.52786 | - | r2=0.9246, rmse=4.372 | ok |
| logreg | taxi | classical2 | 26.7 | 25.9 | sklearn-cpu | 342 | 0.08 | 0.08 |  | accuracy=0.76335, logloss=0.538985, nonfinite_proba_rows=0 | - | accuracy=0.7633, logloss=0.539, nonfinite_proba_rows=0 | ok |
| stacking-reg | istella | algos | 31640 | 11554 | sklearn-cpu | 153421 | 0.21 | 0.08 |  | finite=1, r2=0.448395, rmse=0.620404 | - | r2=0.4476, rmse=0.6208 | ok |
| binarizer | taxi | algos | 0.1 | 0.1 | sklearn-cpu | 1.6 | 0.08 | 0.07 |  | - | - | - | ok |
| normalizer | taxi | algos | 0.1 | 0.1 | sklearn-cpu | 1.6 | 0.07 | 0.07 |  | - | - | - | ok |
| kbins | istella | algos | 1423 | 271 | sklearn-cpu | 3823 | 0.37 | 0.07 |  | - | - | - | ok |
| stacking-clf | istella | algos | 9355 | 9881 | sklearn-cpu | 149799 | 0.06 | 0.07 |  | accuracy=0.92997, logloss=0.193693 | - | accuracy=0.93, logloss=0.1939 | ok |
| hdbscan | istella | classical | 44988 | 45732 | sklearn-cpu | 699145 | 0.06 | 0.07 |  | n_clusters=47, noise_fraction=0.25381, rows=100000 | - | n_clusters=47, noise_fraction=0.2539, rows=1e+05 | ok |
| spectral-embedding | istella | classical2 | 955 | 570 | sklearn-cpu | 8838 | 0.11 | 0.06 |  | trustworthiness_k15=0.783644 | - | trustworthiness_k15=0.8127 | ok |
| mds | taxi | algos | 168 | 169 | sklearn-cpu | 2632 | 0.06 | 0.06 |  | trustworthiness_k15=0.606536 | - | trustworthiness_k15=0.6041 | ok |
| robust-scaler | istella | algos | 1310 | 269 | sklearn-cpu | 4588 | 0.29 | 0.06 |  | - | - | - | ok |
| voting-clf | taxi | algos | 83.4 | 93.4 | sklearn-cpu | 1597 | 0.05 | 0.06 |  | accuracy=0.74229, logloss=0.554797 | - | accuracy=0.7423, logloss=0.5546 | ok |
| ovr | istella | algos | 3896 | 4399 | sklearn-cpu | 75802 | 0.05 | 0.06 |  | accuracy=0.89271 | - | accuracy=0.8927 | ok |
| simple-imputer | taxi | algos | 184 | 21.4 | sklearn-cpu | 370 | 0.50 | 0.06 |  | masked_rmse=5.98518 | - | masked_rmse=5.985 | ok |
| voting-reg | istella | algos | 6096 | 2154 | sklearn-cpu | 37765 | 0.16 | 0.06 |  | finite=1, r2=0.406568, rmse=0.643496 | - | r2=0.4067, rmse=0.6434 | ok |
| mb-sparse-pca | taxi | algos | 161 | 124 | sklearn-cpu | 2223 | 0.07 | 0.06 |  | component_sparsity=0.0227273, relative_reconstruction_error=0.275935 | - | component_sparsity=0.02273, relative_reconstruction_error=0.2759 | ok |
| classical-mds | istella | algos | 176 | 141 | sklearn-cpu | 2623 | 0.07 | 0.05 |  | trustworthiness_k15=0.830548 | - | trustworthiness_k15=0.8306 | ok |
| quantile-transformer | istella | algos | 1424 | 269 | sklearn-cpu | 5049 | 0.28 | 0.05 |  | - | - | - | ok |
| croston | synthetic | algos | 3.1 | 4.7 | statsforecast-cpu | 91.7 | 0.03 | 0.05 |  | forecast_rmse=1.67484 | - | forecast_rmse=1.675 | ok |
| classical-mds | taxi | algos | 157 | 125 | sklearn-cpu | 2535 | 0.06 | 0.05 |  | trustworthiness_k15=0.765649 | - | trustworthiness_k15=0.7657 | ok |
| bagging-reg | istella | algos | 1107 | 1592 | sklearn-cpu | 36039 | 0.03 | 0.04 |  | finite=1, r2=0.523619, rmse=0.576551 | - | r2=0.5185, rmse=0.5797 | ok |
| bagging-clf | istella | algos | 1134 | 1488 | sklearn-cpu | 37699 | 0.03 | 0.04 |  | accuracy=0.9417, logloss=0.149948 | - | accuracy=0.9414, logloss=0.1521 | ok |
| bpe-train | enwik8 | algos | 47.5 | 47.5 | hf-tokenizers-cpu | 1224 | 0.04 | 0.04 |  | jaccard_vs_ours=1, n_tokens=4096 | - | jaccard_vs_ours=0.9995, n_tokens=4096 | ok |
| jl-min-dim | synthetic | algos | 5.4 | 5.7 | sklearn-cpu | 170 | 0.03 | 0.03 |  | - | - | equal_fraction_vs_sklearn=1 | ok |
| decision-tree-clf | taxi | algos | 62.2 | 73.3 | sklearn-cpu | 2218 | 0.03 | 0.03 |  | accuracy=0.7563, logloss=1.20224 | - | accuracy=0.7566, logloss=1.15 | ok |
| decision-tree-reg | taxi | algos | 55.9 | 69.6 | sklearn-cpu | 2117 | 0.03 | 0.03 |  | finite=1, r2=0.862608, rmse=5.90362 | - | r2=0.8915, rmse=5.246 | ok |
| voting-clf | istella | algos | 1192 | 1365 | sklearn-cpu | 47440 | 0.03 | 0.03 |  | accuracy=0.91835, logloss=0.1877 | - | accuracy=0.9185, logloss=0.1875 | ok |
| bootstrap | istella | algos | 15.9 | 13.6 | scipy-cpu | 525 | 0.03 | 0.03 |  | ci_high=0.29485, ci_low=0.27125, standard_error=0.00598712 | - | ci_high=0.2955, ci_low=0.2717, standard_error=0.006044 | ok |
| agglomerative | istella | classical2 | 152 | 124 | sklearn-cpu | 4827 | 0.03 | 0.03 |  | n_clusters=8, silhouette=0.716728 | - | ari_vs_ours=1, n_clusters=8, silhouette=0.7167 | ok |
| radius-neighbors | istella | algos | 0.1 | 0.1 | sklearn-cpu | 5.7 | 0.02 | 0.03 |  | neighbors_total=1.22072e+06 | - | neighbors_total=1.221e+06 | ok |
| poisson | istella | algos | - | 2624 | sklearn-cpu (fill) | 110378 | - | 0.02 |  | finite=1, r2=0.243602, rmse=0.7265 | - | - | ok |
| bootstrap | taxi | algos | 12.3 | 12.2 | scipy-cpu | 525 | 0.02 | 0.02 |  | ci_high=18.7174, ci_low=18.2494, standard_error=0.117837 | - | ci_high=18.71, ci_low=18.25, standard_error=0.1176 | ok |
| select-mutual-info | istella | algos | 797 | 797 | sklearn-cpu | 36427 | 0.02 | 0.02 |  | n_selected=110 | n_selected=110 | jaccard_vs_sklearn=1, n_selected=110 | ok |
| simple-imputer | istella | algos | 1425 | 269 | sklearn-cpu | 12578 | 0.11 | 0.02 |  | masked_rmse=346849 | - | masked_rmse=3.468e+05 | ok |
| permutation-test | istella | algos | 55.3 | 55.4 | scipy-cpu | 3042 | 0.02 | 0.02 |  | pvalue=0.1802, statistic=0.0112 | - | pvalue=0.1844, statistic=0.0112 | ok |
| permutation-test | taxi | algos | 55.3 | 55.6 | scipy-cpu | 3055 | 0.02 | 0.02 |  | pvalue=0.0006, statistic=-0.576702 | - | pvalue=0.001, statistic=-0.5767 | ok |
| gamma | istella | algos | - | 1889 | sklearn-cpu (fill) | 107798 | - | 0.02 |  | finite=1, r2=0.218164, rmse=0.738615 | - | - | ok |
| tweedie | istella | algos | - | 1886 | sklearn-cpu (fill) | 108690 | - | 0.02 |  | finite=1, r2=-24.4125, rmse=4.21099 | - | - | ok |
| huber | istella | algos | 755 | 725 | sklearn-cpu | 45995 | 0.02 | 0.02 |  | finite=1, r2=-0.00719273, rmse=0.838333 | - | r2=-0.01018, rmse=0.8396 | ok |
| adaboost-reg | istella | algos | 12599 | 2347 | sklearn-cpu | 183985 | 0.07 | 0.01 |  | finite=1, r2=0.229908, rmse=0.733047 | - | r2=0.1676, rmse=0.7621 | ok |
| adaboost-clf | istella | algos | 11088 | 7237 | sklearn-cpu (fill) | 615458 | 0.02 | 0.01 |  | accuracy=0.93499, logloss=0.438712 | - | accuracy=0.9371 | ok |
| decision-tree-reg | istella | algos | 438 | 599 | sklearn-cpu | 52136 | 0.01 | 0.01 |  | finite=1, r2=0.379438, rmse=0.658041 | - | r2=0.3732, rmse=0.6613 | ok |
| decision-tree-clf | istella | algos | 449 | 618 | sklearn-cpu | 54800 | 0.01 | 0.01 |  | accuracy=0.935, logloss=0.758303 | - | accuracy=0.9344, logloss=0.7917 | ok |
| power-transformer | taxi | algos | 2698 | 136 | sklearn-cpu | 13228 | 0.20 | 0.01 |  | - | - | - | ok |
| power-transformer | istella | algos | 7060 | 2644 | sklearn-cpu | 276767 | 0.03 | 0.01 |  | - | - | - | ok |
| optics | istella | algos | 450 | 287 | sklearn-cpu | 30143 | 0.01 | 0.01 |  | n_clusters=20, silhouette=-0.287356 | - | ari_vs_ours=0.9846, n_clusters=20, silhouette=-0.2858 | ok |
| normalizer | istella | algos | 0.1 | 0.1 | sklearn-cpu | 25.3 | 0.01 | 0.01 |  | - | - | - | ok |
| binarizer | istella | algos | 0.1 | 0.1 | sklearn-cpu | 25.3 | 0.01 | 0.00 |  | - | - | - | ok |
| meanshift | taxi | algos | 36.2 | 45.3 | sklearn-cpu | 9370 | 0.00 | 0.00 |  | n_clusters=122, silhouette=0.246631 | - | ari_vs_ours=1, n_clusters=122, silhouette=0.2466 | ok |
| radius-neighbors | taxi | algos | 0.1 | 0.1 | sklearn-cpu | 45.0 | 0.00 | 0.00 |  | neighbors_total=31 | - | neighbors_total=31 | ok |
| kde | taxi | classical | 134 | 20.0 | sklearn-cpu | 7001 | 0.02 | 0.00 |  | mean_log_likelihood=-14.8265, rows_without_density=0 | - | mean_log_likelihood=-14.83, rows_without_density=0 | ok |
| quantile | taxi | algos | 496 | 552 | sklearn-cpu | 216369 | 0.00 | 0.00 |  | finite=1, r2=0.899596, rmse=5.04675 | - | r2=0.8997, rmse=5.045 | ok |
| factor-analysis | taxi | algos | 362 | 324 | sklearn-cpu | 144607 | 0.00 | 0.00 |  | mean_log_likelihood=-14.8237 | - | mean_log_likelihood=-14.82 | ok |
| kde | istella | classical | 178 | 91.9 | sklearn-cpu | 54770 | 0.00 | 0.00 |  | mean_log_likelihood=-222.271, rows_without_density=0 | - | mean_log_likelihood=-227, rows_without_density=0 | ok |
| linearsvc | istella | classical2 | 612 | 726 | sklearn-cpu | 613917 | 0.00 | 0.00 |  | accuracy=0.92347 | - | accuracy=0.9235 | ok |
| linearsvr | taxi | classical2 | 41.9 | 36.9 | sklearn-cpu | 85770 | 0.00 | 0.00 |  | finite=1, r2=0.899814, rmse=5.04127 | - | r2=0.8998, rmse=5.042 | ok |
| optics | taxi | algos | 425 | 255 | sklearn-cpu | 767657 | 0.00 | 0.00 |  | n_clusters=127, silhouette=-0.353359 | - | ari_vs_ours=1, n_clusters=127, silhouette=-0.3534 | ok |
| linearsvr | istella | classical2 | 186 | 198 | sklearn-cpu | 710262 | 0.00 | 0.00 |  | finite=1, r2=-0.106761, rmse=0.878794 | - | r2=-0.02573, rmse=0.846 | ok |
| adam | synthetic | algos | - | - | torch-eager-fp32 | 29.3 | - | - |  | - | - | - | MODE-MISMATCH(requested identical, read back unknown) |
| adamw | synthetic | algos | - | - | torch-eager-fp32 | 31.1 | - | - |  | - | - | - | MODE-MISMATCH(requested identical, read back unknown) |
| als | text | algos | - | - | implicit-cpu | 321340 | - | - |  | - | - | recall_at_10=0.5482 | Opponent full measurement; no matching own cell |
| dbscan | istella | classical | - | - | sklearn-cpu | 248709 | - | - |  | - | - | n_clusters=4.013e+04, noise_fraction=0.2194, rows=1e+06 | Opponent full measurement; no matching own cell |
| dbscan | taxi | classical | - | - | - | - | - | - |  | - | - | - | Opponent full measurement; no matching own cell |
| eigh | synthetic | algos | - | - | numpy-cpu | 4813 | - | - |  | - | - | max_eigenvalue_error=3.49e-08, relative_residual=2.824e-08 | REFUSED(timeout: null) |
| elliptic-envelope | istella | algos | - | - | sklearn-cpu | 43759 | - | - |  | - | - | fraction_flagged=0.09157, jaccard_vs_sklearn=1 | REFUSED(timeout: null) |
| embedding | synthetic | algos | - | - | torch-compile-fp32 | 11.1 | - | - |  | - | - | max_rel_diff_vs_torch_eager_fp32=0, rel_fro_vs_torch_eager_fp32=0 | MODE-MISMATCH(requested identical, read back unknown) |
| et | istella | trees | - | - | sklearn-et-cpu | 29943 | - | - |  | - | - | auc=0.9379, logloss=0.1901 | Opponent full measurement; no matching own cell |
| et | taxi | trees | - | - | sklearn-et-cpu | 20451 | - | - |  | - | - | auc=0.619, logloss=0.526 | Opponent full measurement; no matching own cell |
| gamma | istella | algos | - | - | sklearn-cpu | 252506 | - | - |  | - | - | r2=0.3103, rmse=0.693 | Opponent full measurement; no matching own cell |
| gamma | taxi | algos | - | - | sklearn-cpu | 356 | - | - |  | - | - | r2=-233, rmse=243.6 | Opponent full measurement; no matching own cell |
| gbdt-categorical | taxi | trees | - | - | lightgbm-cpu | 59629 | - | - |  | - | - | auc=0.6327, logloss=0.5281 | Opponent full measurement; no matching own cell |
| gbdt-depthwise | istella | trees | - | - | xgboost-cpu | 22718 | - | - |  | - | - | auc=0.9836, logloss=0.1493 | Opponent full measurement; no matching own cell |
| gbdt-depthwise | taxi | trees | - | - | xgboost-cpu | 10444 | - | - |  | - | - | auc=0.631, logloss=0.5287 | Opponent full measurement; no matching own cell |
| gbdt-lossguide | istella | trees | - | - | lightgbm-cpu | 54930 | - | - |  | - | - | auc=0.9838, logloss=0.1497 | Opponent full measurement; no matching own cell |
| gbdt-lossguide | taxi | trees | - | - | lightgbm-cpu | 51689 | - | - |  | - | - | auc=0.6322, logloss=0.5281 | Opponent full measurement; no matching own cell |
| gbdt-multiclass | istella | trees | - | - | xgboost-cpu | 105269 | - | - |  | - | - | accuracy=0.9101, mlogloss=0.2468 | Opponent full measurement; no matching own cell |
| gbdt-multiclass | taxi | trees | - | - | xgboost-cpu | 45617 | - | - |  | - | - | accuracy=0.6011, mlogloss=1.005 | Opponent full measurement; no matching own cell |
| gbdt-ordered | istella | trees | - | - | catboost-cpu | 384495 | - | - |  | - | - | auc=0.9793, logloss=0.1915 | Opponent full measurement; no matching own cell |
| gbdt-ordered | taxi | trees | - | - | catboost-cpu | 98353 | - | - |  | - | - | auc=0.6277, logloss=0.5295 | Opponent full measurement; no matching own cell |
| gbdt-rank-pairlogit | istella | trees | - | - | xgboost-cpu | 8211 | - | - |  | - | - | map=0.8728, ndcg10=0.7384, ndcg5=0.6701 | Opponent full measurement; no matching own cell |
| gbdt-rank-yetirank | istella | trees | - | - | lightgbm-cpu | 6997 | - | - |  | - | - | map=0.8584, ndcg10=0.7415, ndcg5=0.6803 | Opponent full measurement; no matching own cell |
| gbdt-symmetric | istella | trees | - | - | catboost-cpu | 59581 | - | - |  | - | - | auc=0.9799, logloss=0.1881 | Opponent full measurement; no matching own cell |
| gbdt-symmetric | taxi | trees | - | - | catboost-cpu | 28135 | - | - |  | - | - | auc=0.6303, logloss=0.5286 | Opponent full measurement; no matching own cell |
| gbdt-symmetric-1000 | istella | trees | - | - | catboost-cpu | 118951 | - | - |  | - | - | auc=0.9823, logloss=0.1716 | Opponent full measurement; no matching own cell |
| gbdt-symmetric-1000 | taxi | trees | - | - | catboost-cpu | 55079 | - | - |  | - | - | auc=0.6316, logloss=0.5283 | Opponent full measurement; no matching own cell |
| gemm-int8 | gaussian | neural | 297 | 297 | - | - | - | - |  | max_rel_err_vs_fp64=0 | max_rel_err_vs_fp64=0 | - | ok |
| gmm | taxi | classical2 | 604 | 604 | - | - | - | - |  | bic=-3.67e+06, mean_log_likelihood=12.86, n_iter=32 | bic=-3.67e+06, mean_log_likelihood=12.86, n_iter=32 | - | ok |
| iforest | istella | trees | - | - | sklearn-iforest-cpu | 310 | - | - |  | - | - | auc=0.8279 | Opponent full measurement; no matching own cell |
| iforest | taxi | trees | - | - | sklearn-iforest-cpu | 589 | - | - |  | - | - | auc=0.5528 | Opponent full measurement; no matching own cell |
| lamb | synthetic | algos | - | - | - | - | - | - |  | - | - | - | REFUSED(error: {"error": "Exception('lamb_step: offsets must rise strictly from 0, below 2^24')", "event": "error", "sta |
| lars | istella | algos | 442 | 380 | - | - | - | - |  | finite=1, r2=0.309043, rmse=0.694362 | - | - | ok |
| lion | synthetic | algos | 135 | 135 | - | - | - | - |  | - | - | - | ok |
| min-cov-det | istella | algos | - | - | sklearn-cpu | 43632 | - | - |  | - | - | n_features=220 | REFUSED(timeout: null) |
| poisson | istella | algos | - | - | sklearn-cpu | 112432 | - | - |  | - | - | r2=0.2513, rmse=0.7228 | Opponent full measurement; no matching own cell |
| poisson | taxi | algos | - | - | sklearn-cpu | 430 | - | - |  | - | - | r2=0.03621, rmse=15.64 | Opponent full measurement; no matching own cell |
| qn-reg | istella | algos | 371 | 437 | - | - | - | - |  | finite=1, r2=0.327491, rmse=0.68503 | - | - | ok |
| qn-reg | istella | algos | - | - | sklearn-cpu | 3222 | - | - |  | - | - | r2=0.2553, rmse=0.7201 | Opponent full measurement; no matching own cell |
| qn-reg | taxi | algos | 18.2 | 21.1 | - | - | - | - |  | finite=1, r2=0.908983, rmse=4.80504 | - | - | ok |
| qn-reg | taxi | algos | - | - | sklearn-cpu | 322 | - | - |  | - | - | r2=0.823, rmse=6.544 | Opponent full measurement; no matching own cell |
| quantile | istella | algos | - | 806 | sklearn-cpu: too slow to measure (216 s on taxi) | - | - | - |  | finite=1, r2=-0.0399981, rmse=0.851877 | - | - | ok |
| rf | istella | trees | - | - | lightgbm-cpu | 67953 | - | - |  | - | - | auc=0.9454, logloss=0.1954 | Opponent full measurement; no matching own cell |
| rf | taxi | trees | - | - | lightgbm-cpu | 54952 | - | - |  | - | - | auc=0.617, logloss=0.5264 | Opponent full measurement; no matching own cell |
| sgd | synthetic | algos | - | - | torch-eager-fp32 | 17.9 | - | - |  | - | - | - | MODE-MISMATCH(requested identical, read back unknown) |
| sgd-clf | istella | algos | - | - | sklearn-cpu | 28859 | - | - |  | - | - | accuracy=0.9102 | Opponent full measurement; no matching own cell |
| sgd-clf | taxi | algos | - | - | sklearn-cpu | 5439 | - | - |  | - | - | accuracy=0.7525 | Opponent full measurement; no matching own cell |
| sgd-reg | istella | algos | - | - | sklearn-cpu | 47955 | - | - |  | - | - | r2=-2.196e+24, rmse=1.238e+12 | Opponent full measurement; no matching own cell |
| sgd-reg | taxi | algos | - | - | sklearn-cpu | 5323 | - | - |  | - | - | r2=0.8807, rmse=5.502 | Opponent full measurement; no matching own cell |
| tweedie | istella | algos | - | - | sklearn-cpu | 250154 | - | - |  | - | - | r2=-4.58e+04, rmse=178.6 | Opponent full measurement; no matching own cell |
| tweedie | taxi | algos | - | - | sklearn-cpu | 4055 | - | - |  | - | - | r2=-25.82, rmse=80.55 | Opponent full measurement; no matching own cell |

Sources: before = M3 0.8.34 board (classical), M3 2026-09-29 board IDENTICAL cells (trees); job tags ident-adaboost-clf-istella, ident-adaboost-clf-taxi, ident-adaboost-reg-istella, ident-adaboost-reg-taxi, ident-additive-chi2-istella, ident-additive-chi2-taxi, ident-affinity-prop-istella, ident-affinity-prop-taxi, ident-agglomerative-istella, ident-agglomerative-taxi, ident-als-taxi-zones, ident-als-text, ident-ard-istella, ident-ard-taxi, ident-arima-synthetic, ident-auto-theta-synthetic, ident-auto-theta-taxi-hourly, ident-autoarima-synthetic, ident-autoarima-taxi-hourly, ident-bagging-clf-istella, ident-bagging-clf-taxi, ident-bagging-reg-istella, ident-bagging-reg-taxi, ident-bayesian-gmm-istella, ident-bayesian-gmm-taxi, ident-bayesian-ridge-istella, ident-bayesian-ridge-taxi, ident-bernoulli-nb-istella, ident-bernoulli-nb-taxi, ident-binarizer-istella, ident-binarizer-taxi, ident-bisecting-kmeans-istella, ident-bisecting-kmeans-taxi, ident-bootstrap-istella, ident-bootstrap-taxi, ident-bpe-encode-enwik8, ident-bpe-train-enwik8, ident-cagra-istella, ident-cagra-taxi, ident-calibrated-istella, ident-calibrated-taxi, ident-categorical-nb-istella, ident-categorical-nb-taxi, ident-cca-istella, ident-cca-taxi, ident-cholesky-synthetic, ident-classical-mds-istella, ident-classical-mds-taxi, ident-complement-nb-istella, ident-complement-nb-taxi, ident-complement-nb-text, ident-connected-components-istella, ident-connected-components-taxi, ident-cross-val-score-istella, ident-cross-val-score-taxi, ident-croston-optimized-synthetic, ident-croston-optimized-taxi-hourly, ident-croston-sba-synthetic, ident-croston-sba-taxi-hourly, ident-croston-synthetic, ident-croston-taxi-hourly, ident-damped-ets-synthetic, ident-damped-ets-taxi-hourly, ident-dart-istella, ident-dart-reg-istella, ident-dart-reg-taxi, ident-dart-taxi, ident-decision-tree-clf-istella, ident-decision-tree-clf-taxi, ident-decision-tree-reg-istella, ident-decision-tree-reg-taxi, ident-dict-learning-istella, ident-dict-learning-taxi, ident-dynamic-optimized-theta-synthetic, ident-dynamic-optimized-theta-taxi-hourly, ident-dynamic-theta-synthetic, ident-dynamic-theta-taxi-hourly, ident-elliptic-envelope-taxi, ident-enet-cv-istella, ident-enet-cv-taxi, ident-ets-synthetic, ident-factor-analysis-istella, ident-factor-analysis-taxi, ident-fastica-istella, ident-fastica-taxi, ident-gamma-istella, ident-gamma-taxi, ident-garch-synthetic, ident-garch-taxi-hourly, ident-gaussian-nb-istella, ident-gaussian-nb-taxi, ident-gaussian-rp-istella, ident-gaussian-rp-taxi, ident-gmm-istella, ident-gpr-istella, ident-gpr-taxi, ident-hdbscan-istella, ident-hdbscan-taxi, ident-huber-istella, ident-huber-taxi, ident-incremental-pca-istella, ident-incremental-pca-taxi, ident-isomap-istella, ident-isomap-taxi, ident-isotonic-istella, ident-isotonic-taxi, ident-iterative-imputer-istella, ident-iterative-imputer-taxi, ident-ivf-filter-istella, ident-ivf-filter-taxi, ident-ivf-istella, ident-ivf-pq-istella, ident-ivf-pq-taxi, ident-ivf-rabitq-istella, ident-ivf-rabitq-taxi, ident-ivf-refine-istella, ident-ivf-sq-istella, ident-ivf-sq-taxi, ident-ivf-taxi, ident-jl-min-dim-synthetic, ident-kbins-istella, ident-kbins-taxi, ident-kde-istella, ident-kde-taxi, ident-kernel-pca-istella, ident-kernel-pca-taxi, ident-kernel-ridge-istella, ident-kernel-ridge-taxi, ident-kernel-shap-istella, ident-kernel-shap-taxi, ident-kmeans-istella, ident-kmeans-taxi, ident-knn-clf-istella, ident-knn-clf-taxi, ident-knn-imputer-istella, ident-knn-imputer-taxi, ident-knn-istella, ident-knn-reg-istella, ident-knn-reg-taxi, ident-knn-taxi, ident-kpss-synthetic, ident-kpss-taxi-hourly, ident-label-binarizer-istella, ident-label-binarizer-taxi, ident-label-encoder-istella, ident-label-encoder-taxi, ident-label-propagation-istella, ident-label-propagation-taxi, ident-label-spreading-istella, ident-label-spreading-taxi, ident-lars-istella, ident-lars-taxi, ident-lasso-cv-istella, ident-lasso-cv-taxi, ident-lasso-lars-istella, ident-lasso-lars-taxi, ident-lda-clf-istella, ident-lda-clf-taxi, ident-lda-taxi-zones, ident-lda-text, ident-linearsvc-istella, ident-linearsvc-taxi, ident-linearsvr-istella, ident-linearsvr-taxi, ident-lle-istella, ident-lle-taxi, ident-lof-istella, ident-lof-taxi, ident-logreg-cv-istella, ident-logreg-cv-taxi, ident-logreg-istella, ident-logreg-taxi, ident-louvain-istella, ident-louvain-taxi, ident-lstsq-istella, ident-lstsq-taxi, ident-lu-factor-synthetic, ident-lu-solve-synthetic, ident-maxabs-scaler-istella, ident-maxabs-scaler-taxi, ident-mb-dict-learning-istella, ident-mb-dict-learning-taxi, ident-mb-sparse-pca-istella, ident-mb-sparse-pca-taxi, ident-mds-istella, ident-mds-taxi, ident-meanshift-istella, ident-meanshift-taxi, ident-min-cov-det-taxi, ident-minibatch-kmeans-istella, ident-minibatch-kmeans-taxi, ident-minmax-scaler-istella, ident-minmax-scaler-taxi, ident-multilabel-binarizer-istella, ident-multilabel-binarizer-taxi, ident-multinomial-nb-istella, ident-multinomial-nb-taxi, ident-multinomial-nb-text, ident-multioutput-clf-istella, ident-multioutput-clf-taxi, ident-multioutput-reg-istella, ident-multioutput-reg-taxi, ident-nearest-centroid-istella, ident-nearest-centroid-taxi, ident-nmf-istella, ident-nmf-taxi, ident-normalizer-istella, ident-normalizer-taxi, ident-nystroem-istella, ident-nystroem-taxi, ident-ocsvm-istella, ident-ocsvm-taxi, ident-ols-istella, ident-ols-taxi, ident-onehot-istella, ident-onehot-taxi, ident-optics-istella, ident-optics-taxi, ident-optimized-theta-synthetic, ident-optimized-theta-taxi-hourly, ident-ordinal-istella, ident-ordinal-taxi, ident-ovr-istella, ident-ovr-taxi, ident-pa-clf-istella, ident-pa-clf-taxi, ident-pa-reg-istella, ident-pa-reg-taxi, ident-pagerank-istella, ident-pagerank-taxi, ident-pca-istella, ident-pca-taxi, ident-perceptron-istella, ident-perceptron-taxi, ident-permutation-shap-istella, ident-permutation-shap-taxi, ident-permutation-test-istella, ident-permutation-test-taxi, ident-pls-canonical-istella, ident-pls-canonical-taxi, ident-pls-istella, ident-pls-taxi, ident-poisson-istella, ident-poisson-taxi, ident-poly-count-sketch-istella, ident-poly-count-sketch-taxi, ident-poly-features-istella, ident-poly-features-taxi, ident-power-transformer-istella, ident-power-transformer-taxi, ident-prophet-synthetic, ident-prophet-taxi-hourly, ident-qda-istella, ident-qda-taxi, ident-qn-reg-istella, ident-qn-reg-taxi, ident-qr-istella, ident-qr-taxi, ident-quantile-istella, ident-quantile-taxi, ident-quantile-transformer-istella, ident-quantile-transformer-taxi, ident-radius-neighbors-istella, ident-radius-neighbors-taxi, ident-random-trees-embedding-istella, ident-random-trees-embedding-taxi, ident-randomized-svd-istella, ident-randomized-svd-taxi, ident-rbf-sampler-istella, ident-rbf-sampler-taxi, ident-resample-istella, ident-resample-taxi, ident-rfe-istella, ident-rfe-taxi, ident-ridge-clf-istella, ident-ridge-clf-taxi, ident-ridge-cv-istella, ident-ridge-cv-taxi, ident-ridge-istella, ident-ridge-taxi, ident-robust-scaler-istella, ident-robust-scaler-taxi, ident-select-chi2-istella, ident-select-chi2-taxi, ident-select-d-synthetic, ident-select-d-taxi-hourly, ident-select-f-classif-istella, ident-select-f-classif-taxi, ident-select-f-regression-istella, ident-select-f-regression-taxi, ident-select-mutual-info-reg-taxi, ident-select-mutual-info-taxi, ident-select-r-regression-istella, ident-select-r-regression-taxi, ident-sgd-ocsvm-istella, ident-sgd-ocsvm-taxi, ident-simple-imputer-istella, ident-simple-imputer-taxi, ident-skewed-chi2-istella, ident-skewed-chi2-taxi, ident-sparse-coder-istella, ident-sparse-coder-taxi, ident-sparse-pca-istella, ident-sparse-pca-taxi, ident-sparse-rp-istella, ident-sparse-rp-taxi, ident-spectral-embedding-istella, ident-spectral-embedding-taxi, ident-spectral-istella, ident-spectral-taxi, ident-spline-istella, ident-spline-taxi, ident-stacking-clf-istella, ident-stacking-clf-taxi, ident-stacking-reg-istella, ident-stacking-reg-taxi, ident-standard-scaler-istella, ident-standard-scaler-taxi, ident-stl-synthetic, ident-stl-taxi-hourly, ident-svc-istella, ident-svc-taxi, ident-svd-istella, ident-svd-taxi, ident-svgp-istella, ident-svgp-taxi, ident-svr-istella, ident-svr-taxi, ident-target-encoder-istella, ident-target-encoder-taxi, ident-theta-synthetic, ident-theta-taxi-hourly, ident-tree-shap-istella, ident-tree-shap-taxi, ident-tsne-istella, ident-tsne-taxi, ident-tsvd-istella, ident-tsvd-taxi, ident-tweedie-istella, ident-tweedie-taxi, ident-umap-istella, ident-umap-taxi, ident-var-synthetic, ident-var-taxi-hourly, ident-variance-threshold-istella, ident-variance-threshold-taxi, ident-voting-clf-istella, ident-voting-clf-taxi, ident-voting-reg-istella, ident-voting-reg-taxi.

- CPU resource policy: CPU opponents in the 2026-10-05 fresh sweep received the full 28-core M3 Ultra allocation with no imposed CPU thread cap. Library parallelism varies; all-core availability does not mean all cores were busy.
- CPU resource policy: Stored historical opponent cells keep their original resource provenance. Fresh measurements use one excluded warmup and one scored sample; failures remain failures.
- CPU resource policy: At the user-directed transition to missing-only opponents, the active LinearSVR/Istella attempt was interrupted without admitting a timing. Completed measurements were preserved; subsequent captures are recorded under opponent_imports.
